# -*- coding:utf-8 -*-

"""
    把 npz权重文件（.pth模型转换而成）, 转成 tensorrt 序列化模型文件
    并使用 tensorrt runtime 推理
"""

import os
import time

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

import calibrator
from mask2color import to_color

INPUT_NAME = "data"
OUTPUT_NAME = "prob"
classes_num = 32
input_height = 448
input_width = 448
para_file = "./para.npz"
trt_file = "./model.plan"
data_path = "../../../../Camvid_segment_dataset"
val_data_path = data_path + "/images/val"  # 用于 int8 量化
test_data_path = data_path + "/images/test"  # 用于推理

# for FP16 mode
use_fp16_mode = False
# for INT8 model
use_int8_mode = False
n_calibration = 20
cache_file = "./int8.cache"
calibration_data_path = val_data_path

np.set_printoptions(precision=3, linewidth=160, suppress=True)


def add_batch_norm_2d(network, para, layer_name, input_tensor, eps=1e-5):
    # reference: https://blog.csdn.net/wq_0708/article/details/121400682
    gamma = np.ascontiguousarray(para[layer_name + ".weight"])
    beta = np.ascontiguousarray(para[layer_name + ".bias"])
    mean = np.ascontiguousarray(para[layer_name + ".running_mean"])
    var = np.ascontiguousarray(para[layer_name + ".running_var"])

    scale = trt.Weights(gamma / np.sqrt(var + eps))
    shift = trt.Weights(beta - mean / np.sqrt(var + eps) * gamma)
    power = trt.Weights(np.ones(len(var), dtype=np.float32))

    scale_layer = network.add_scale(input_tensor, trt.ScaleMode.CHANNEL, shift, scale, power)

    return scale_layer


def double_conv(network, para, input_tensor, outch: int, layer_name: str):
    w = np.ascontiguousarray(para[layer_name + ".double_conv.0.weight"])
    # b = np.zeros(w.shape[0], dtype=np.float32)
    conv1 = network.add_convolution_nd(input_tensor, outch, [3, 3], trt.Weights(w), trt.Weights())
    conv1.stride_nd = [1, 1]
    conv1.padding_nd = [1, 1]

    bn1 = add_batch_norm_2d(network, para, layer_name + ".double_conv.1", conv1.get_output(0))

    relu1 = network.add_activation(bn1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para[layer_name + ".double_conv.3.weight"])
    # b = np.zeros(w.shape[0], dtype=np.float32)
    conv2 = network.add_convolution_nd(relu1.get_output(0), outch, [3, 3], trt.Weights(w), trt.Weights())
    conv2.stride_nd = [1, 1]
    conv2.padding_nd = [1, 1]

    bn2 = add_batch_norm_2d(network, para, layer_name + ".double_conv.4", conv2.get_output(0))

    relu2 = network.add_activation(bn2.get_output(0), trt.ActivationType.RELU)

    return relu2


def down(network, para, input_tensor, outch: int, layer_name: str):
    pool1 = network.add_pooling_nd(input_tensor, trt.PoolingType.MAX, [2, 2])
    pool1.stride_nd = [2, 2]

    return double_conv(network, para, pool1.get_output(0), outch, layer_name + ".maxpool_conv.1")


def up(network, para, input_tensor1, input_tensor2, inch: int, outch: int, layer_name: str):
    w = np.ascontiguousarray(para[layer_name + ".up.weight"])
    b = np.ascontiguousarray(para[layer_name + ".up.bias"])
    deconv = network.add_deconvolution_nd(input_tensor1, inch // 2, [2, 2], trt.Weights(w), trt.Weights(b))
    deconv.stride_nd = (2, 2)

    diffy = input_tensor2.shape[2] - deconv.get_output(0).shape[2]
    diffx = input_tensor2.shape[3] - deconv.get_output(0).shape[3]
    # pad = network.add_padding_nd(deconv.get_output(0), (diffy // 2, diffx // 2), (diffy - diffy // 2, diffx - diffx // 2))
    output_shape = (1, inch // 2, deconv.get_output(0).shape[2] + diffy, deconv.get_output(0).shape[3] + diffx)
    pad = network.add_slice(deconv.get_output(0), (0, 0, - diffy // 2, - diffx // 2), output_shape, (1, 1, 1, 1))
    pad.mode = trt.SliceMode.FILL

    concat = network.add_concatenation([input_tensor2, pad.get_output(0)])
    concat.axis = 1

    return double_conv(network, para, concat.get_output(0), outch, layer_name + ".conv")


def build_network(network, profile, config):
    input_tensor = network.add_input(INPUT_NAME, trt.float32, [-1, 3, input_height, input_width])
    profile.set_shape(input_tensor.name, [1, 3, input_height, input_width], [4, 3, input_height, input_width],
                      [8, 3, input_height, input_width])
    config.add_optimization_profile(profile)

    para = np.load(para_file)

    inc = double_conv(network, para, input_tensor, 64, "inc")
    down1 = down(network, para, inc.get_output(0), 128, "down1")
    down2 = down(network, para, down1.get_output(0), 256, "down2")
    down3 = down(network, para, down2.get_output(0), 512, "down3")
    down4 = down(network, para, down3.get_output(0), 1024, "down4")
    up1 = up(network, para, down4.get_output(0), down3.get_output(0), 1024, 512, "up1")
    up2 = up(network, para, up1.get_output(0), down2.get_output(0), 512, 256, "up2")
    up3 = up(network, para, up2.get_output(0), down1.get_output(0), 256, 128, "up3")
    up4 = up(network, para, up3.get_output(0), inc.get_output(0), 128, 64, "up4")
    # output conv
    w = np.ascontiguousarray(para["outc.conv.weight"])
    b = np.ascontiguousarray(para["outc.conv.bias"])
    out_conv = network.add_convolution_nd(up4.get_output(0), classes_num, [1, 1], trt.Weights(w), trt.Weights(b))
    # add topk layer
    top1_layer = network.add_topk(out_conv.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)
    top1_layer.get_output(1).name = OUTPUT_NAME
    network.mark_output(top1_layer.get_output(1))


def get_engine():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.exists(trt_file):
        with open(trt_file, "rb") as f:  # read .plan file if exists
            engine_string = f.read()
        if engine_string is None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # set workspace for TensorRT
        if use_fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        if use_int8_mode:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator.MyCalibrator(calibration_data_path, n_calibration,
                                                             (8, 3, input_height, input_width), cache_file)

        build_network(network, profile, config)

        engine_string = builder.build_serialized_network(network, config)
        if engine_string is None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trt_file, "wb") as f:
            f.write(engine_string)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)

    return engine


def image_preprocess(np_img):
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)  # bgr to rgb
    # resize
    img = cv2.resize(img, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    # normalize
    data = img.astype(np.float32)
    data = data / 255.
    # transpose
    data = data.transpose((2, 0, 1)).astype(np.float32)  # HWC to CHW

    return data


def image_postprocess(mask, origin_h, origin_w, file_name):
    # resize
    resized_mask = cv2.resize(mask.astype(np.uint8), (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
    # blur
    blur_mask = cv2.medianBlur(resized_mask, 7)
    # to color
    color_img = to_color(blur_mask)
    # rgb to bgr
    bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    # save
    cv2.imwrite(file_name.split(".")[0] + "_mask.png", bgr)


def inference_one(data_input, context, buffer_h, buffer_d):
    """
        使用tensorrt runtime 做一次推理
    :param data_input: 经过预处理（缩放、裁剪、归一化等）后的图像数据（ndarray类型）
    """
    buffer_h[0] = np.ascontiguousarray(data_input)
    cudart.cudaMemcpy(buffer_d[0], buffer_h[0].ctypes.data, buffer_h[0].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(buffer_d)  # inference

    cudart.cudaMemcpy(buffer_h[1].ctypes.data, buffer_d[1], buffer_h[1].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outs = buffer_h[-1].reshape((input_height, input_width))

    return outs


if __name__ == '__main__':
    engine = get_engine()

    n_io = engine.num_bindings
    l_tensor_name = [engine.get_binding_name(i) for i in range(n_io)]
    n_input = np.sum([engine.binding_is_input(i) for i in range(n_io)])

    context = engine.create_execution_context()
    context.set_binding_shape(0, [1, 3, input_height, input_width])
    for i in range(n_io):
        print("[%2d]%s->" % (i, "Input " if i < n_input else "Output"), engine.get_binding_dtype(i),
              engine.get_binding_shape(i), context.get_binding_shape(i), l_tensor_name[i])

    buffer_h = []
    for i in range(n_io):
        buffer_h.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    buffer_d = []
    for i in range(n_io):
        buffer_d.append(cudart.cudaMalloc(buffer_h[i].nbytes)[1])

    total_cost = 0
    img_count = 0
    for image_name in os.listdir(test_data_path):
        image_path = os.path.join(test_data_path, image_name)
        if image_path.endswith("jpg") or image_path.endswith("jpeg"):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # read image
            origin_height, origin_width = image.shape[:2]  # get origin resolution

            start = time.time()
            input_data = image_preprocess(image)  # image preprocess
            input_data = np.expand_dims(input_data, axis=0)  # add batch size dimension

            output = inference_one(input_data, context, buffer_h, buffer_d)
            end = time.time()

            image_postprocess(output, origin_height, origin_width, image_name)  # for visualization, not necessary

            total_cost += end - start
            img_count += 1

    print("Total image num is: %d, inference total cost is: %.3f, average cost is: %.3f" % (
        img_count, total_cost, total_cost / img_count))

    for bd in buffer_d:
        cudart.cudaFree(bd)
