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
OUTPUT_NAME = "mask"
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


def bottleneck(network, para, input_tensor, inch: int, outch: int, stride: int, dilation: int, layer_name: str):
    w = np.ascontiguousarray(para[layer_name + ".conv1.weight"])
    conv1 = network.add_convolution_nd(input_tensor, outch, [1, 1], trt.Weights(w), trt.Weights())
    bn1 = add_batch_norm_2d(network, para, layer_name + ".bn1", conv1.get_output(0))
    relu1 = network.add_activation(bn1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para[layer_name + ".conv2.weight"])
    conv2 = network.add_convolution_nd(relu1.get_output(0), outch, [3, 3], trt.Weights(w), trt.Weights())
    conv2.dilation_nd = [dilation, dilation]
    conv2.padding_nd = [dilation, dilation]
    conv2.stride_nd = [stride, stride]
    bn2 = add_batch_norm_2d(network, para, layer_name + ".bn2", conv2.get_output(0))
    relu2 = network.add_activation(bn2.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para[layer_name + ".conv3.weight"])
    conv3 = network.add_convolution_nd(relu2.get_output(0), outch * 4, [1, 1], trt.Weights(w), trt.Weights())
    bn3 = add_batch_norm_2d(network, para, layer_name + ".bn3", conv3.get_output(0))

    if stride != 1 or inch != outch * 4:
        w = np.ascontiguousarray(para[layer_name + ".downsample.0.weight"])
        conv4 = network.add_convolution_nd(input_tensor, outch * 4, [1, 1], trt.Weights(w), trt.Weights())
        conv4.stride_nd = [stride, stride]

        bn4 = add_batch_norm_2d(network, para, layer_name + ".downsample.1", conv4.get_output(0))

        ew1 = network.add_elementwise(bn4.get_output(0), bn3.get_output(0), trt.ElementWiseOperation.SUM)
    else:
        ew1 = network.add_elementwise(input_tensor, bn3.get_output(0), trt.ElementWiseOperation.SUM)

    relu3 = network.add_activation(ew1.get_output(0), trt.ActivationType.RELU)

    return relu3


def build_backbone(network, para, input_tensor):
    w = np.ascontiguousarray(para["backbone.conv1.weight"])
    conv1 = network.add_convolution_nd(input_tensor, 64, [7, 7], trt.Weights(w), trt.Weights())
    conv1.stride_nd = [2, 2]
    conv1.padding_nd = [3, 3]

    bn1 = add_batch_norm_2d(network, para, "backbone.bn1", conv1.get_output(0))

    relu1 = network.add_activation(bn1.get_output(0), trt.ActivationType.RELU)

    pool1 = network.add_pooling_nd(relu1.get_output(0), trt.PoolingType.MAX, [3, 3])
    pool1.stride_nd = [2, 2]
    pool1.padding_nd = [1, 1]

    x = bottleneck(network, para, pool1.get_output(0), 64, 64, 1, 1, "backbone.layer1.0")
    x = bottleneck(network, para, x.get_output(0), 256, 64, 1, 1, "backbone.layer1.1")
    x = bottleneck(network, para, x.get_output(0), 256, 64, 1, 1, "backbone.layer1.2")
    low_level_layer = x

    x = bottleneck(network, para, x.get_output(0), 256, 128, 2, 1, "backbone.layer2.0")
    x = bottleneck(network, para, x.get_output(0), 512, 128, 1, 1, "backbone.layer2.1")
    x = bottleneck(network, para, x.get_output(0), 512, 128, 1, 1, "backbone.layer2.2")
    x = bottleneck(network, para, x.get_output(0), 512, 128, 1, 1, "backbone.layer2.3")

    x = bottleneck(network, para, x.get_output(0), 512, 256, 2, 1, "backbone.layer3.0")
    for i in range(1, 6):
        x = bottleneck(network, para, x.get_output(0), 1024, 256, 1, 1, "backbone.layer3." + str(i))

    x = bottleneck(network, para, x.get_output(0), 1024, 512, 1, 2, "backbone.layer4.0")
    x = bottleneck(network, para, x.get_output(0), 2048, 512, 1, 4, "backbone.layer4.1")
    x = bottleneck(network, para, x.get_output(0), 2048, 512, 1, 8, "backbone.layer4.2")

    return x, low_level_layer


def sub_aspp(network, para, input_tensor, kernel_size: int, padding: int, dilation: int, layer_name: str):
    w = np.ascontiguousarray(para[layer_name + ".atrous_conv.weight"])
    conv = network.add_convolution_nd(input_tensor, 256, [kernel_size, kernel_size], trt.Weights(w), trt.Weights())
    conv.dilation_nd = [dilation, dilation]
    conv.padding_nd = [padding, padding]
    conv.stride_nd = [1, 1]

    bn = add_batch_norm_2d(network, para, layer_name + ".bn", conv.get_output(0))

    relu = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)

    return relu


def global_avg_pool(network, para, input_tensor, layer_name):
    pool = network.add_pooling_nd(input_tensor, trt.PoolingType.AVERAGE, input_tensor.shape[2:])
    w = np.ascontiguousarray(para[layer_name + ".1.weight"])
    conv = network.add_convolution_nd(pool.get_output(0), 256, [1, 1], trt.Weights(w), trt.Weights())
    conv.stride_nd = [1, 1]

    bn = add_batch_norm_2d(network, para, layer_name + ".2", conv.get_output(0))

    relu = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)

    return relu


def build_aspp(network, para, input_tensor):
    aspp1 = sub_aspp(network, para, input_tensor, 1, 0, 1, "aspp.aspp1")
    aspp2 = sub_aspp(network, para, input_tensor, 3, 6, 6, "aspp.aspp2")
    aspp3 = sub_aspp(network, para, input_tensor, 3, 12, 12, "aspp.aspp3")
    aspp4 = sub_aspp(network, para, input_tensor, 3, 18, 18, "aspp.aspp4")

    gap = global_avg_pool(network, para, input_tensor, "aspp.global_avg_pool")
    gap_rsz = network.add_resize(gap.get_output(0))
    gap_rsz.shape = (1, 256, aspp4.get_output(0).shape[2], aspp4.get_output(0).shape[3])
    gap_rsz.resize_mode = trt.ResizeMode.LINEAR  # 指定插值方法，默认值 trt.ResizeMode.NEAREST
    gap_rsz.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS

    concat = network.add_concatenation(
        [aspp1.get_output(0), aspp2.get_output(0), aspp3.get_output(0), aspp4.get_output(0), gap_rsz.get_output(0)])
    concat.axis = 1

    w = np.ascontiguousarray(para["aspp.conv1.weight"])
    conv = network.add_convolution_nd(concat.get_output(0), 256, [1, 1], trt.Weights(w), trt.Weights())
    bn = add_batch_norm_2d(network, para, "aspp.bn1", conv.get_output(0))
    relu = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)

    return relu


def build_decoder(network, para, input_tensor1, input_tensor2):
    w = np.ascontiguousarray(para["decoder.conv1.weight"])
    conv1 = network.add_convolution_nd(input_tensor2, 48, [1, 1], trt.Weights(w), trt.Weights())
    bn1 = add_batch_norm_2d(network, para, "decoder.bn1", conv1.get_output(0))
    relu1 = network.add_activation(bn1.get_output(0), trt.ActivationType.RELU)

    rsz = network.add_resize(input_tensor1)
    rsz.shape = (1, 256, relu1.get_output(0).shape[2], relu1.get_output(0).shape[3])
    rsz.resize_mode = trt.ResizeMode.LINEAR  # 指定插值方法，默认值 trt.ResizeMode.NEAREST
    rsz.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS

    concat = network.add_concatenation([rsz.get_output(0), relu1.get_output(0)])
    concat.axis = 1

    w = np.ascontiguousarray(para["decoder.last_conv.0.weight"])
    conv2 = network.add_convolution_nd(concat.get_output(0), 256, [3, 3], trt.Weights(w), trt.Weights())
    conv2.padding_nd = [1, 1]
    conv2.stride_nd = [1, 1]
    bn2 = add_batch_norm_2d(network, para, "decoder.last_conv.1", conv2.get_output(0))
    relu2 = network.add_activation(bn2.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["decoder.last_conv.4.weight"])
    conv3 = network.add_convolution_nd(relu2.get_output(0), 256, [3, 3], trt.Weights(w), trt.Weights())
    conv3.padding_nd = [1, 1]
    conv3.stride_nd = [1, 1]
    bn3 = add_batch_norm_2d(network, para, "decoder.last_conv.5", conv3.get_output(0))
    relu3 = network.add_activation(bn3.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["decoder.last_conv.8.weight"])
    b = np.ascontiguousarray(para["decoder.last_conv.8.bias"])
    conv4 = network.add_convolution_nd(relu3.get_output(0), classes_num, [1, 1], trt.Weights(w), trt.Weights(b))

    return conv4


def build_network(network, profile, config):
    input_tensor = network.add_input(INPUT_NAME, trt.float32, [-1, 3, input_height, input_width])
    profile.set_shape(input_tensor.name, [1, 3, input_height, input_width], [4, 3, input_height, input_width],
                      [8, 3, input_height, input_width])
    config.add_optimization_profile(profile)

    para = np.load(para_file)

    # build backbone
    backbone_out, low_level_layer = build_backbone(network, para, input_tensor)
    print("Backbone built, output shape is :", end=" ")
    print(backbone_out.get_output(0).shape, end=", ")  # (-1, 2048, 28, 28)
    print("low_level_layer shape is:", end=" ")
    print(low_level_layer.get_output(0).shape)  # (-1, 256, 112, 112)

    # build aspp
    aspp = build_aspp(network, para, backbone_out.get_output(0))
    print("ASPP built, output shape is :", end=" ")
    print(aspp.get_output(0).shape)  # (1, 256, 28, 28)

    # build decoder
    decoder = build_decoder(network, para, aspp.get_output(0), low_level_layer.get_output(0))
    print("Decoder built, output shape is :", end=" ")
    print(decoder.get_output(0).shape)

    # resize to origin shape
    rsz = network.add_resize(decoder.get_output(0))
    rsz.shape = (1, classes_num, input_height, input_width)
    rsz.resize_mode = trt.ResizeMode.LINEAR  # 指定插值方法，默认值 trt.ResizeMode.NEAREST
    rsz.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS

    # add topk layer
    top1_layer = network.add_topk(rsz.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)
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
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data = img.astype(np.float32)
    data = (data / 255. - np.array(mean)) / np.array(std)
    # transpose
    data = data.transpose((2, 0, 1)).astype(np.float32)  # HWC to CHW

    return data


def image_postprocess(mask, origin_h, origin_w, file_name):
    # resize
    resized_mask = cv2.resize(mask.astype(np.uint8), (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
    # blur
    blur_mask = cv2.medianBlur(resized_mask, 3)
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
