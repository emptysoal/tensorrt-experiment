# -*- coding: utf-8 -*-

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

INPUT_NAME = "data"
OUTPUT_NAME = "prob"
classes_num = 5
index2class_name = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}
input_size = (224, 224)  # (rows, cols)
para_file = "./para.npz"
trt_file = "./model.plan"
data_path = "../../../../flower_classify_dataset"
val_data_path = data_path + "/val"  # 用于 int8 量化
test_data_path = data_path + "/test"  # 用于推理

# for FP16 mode
use_fp16_mode = False
# for INT8 model
use_int8_mode = False
n_calibration = 20
cache_file = "./int8.cache"
calibration_data_path = val_data_path

np.set_printoptions(precision=3, linewidth=160, suppress=True)

logger = trt.Logger(trt.Logger.ERROR)


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


def build_network(network, profile, config):
    input_tensor = network.add_input(INPUT_NAME, trt.float32, [-1, 3, input_size[0], input_size[1]])
    profile.set_shape(input_tensor.name, [1, 3, input_size[0], input_size[1]], [4, 3, input_size[0], input_size[1]],
                      [8, 3, input_size[0], input_size[1]])
    config.add_optimization_profile(profile)

    para = np.load(para_file)

    # 第一层，2个卷积层和一个最大池化层
    w = np.ascontiguousarray(para["conv1.0.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv1_conv2d_1 = network.add_convolution_nd(input_tensor, 64, [3, 3], trt.Weights(w), trt.Weights(b))
    conv1_conv2d_1.padding_nd = [1, 1]
    conv1_conv2d_1.stride_nd = [1, 1]

    conv1_bn2d_1 = add_batch_norm_2d(network, para, "conv1.1", conv1_conv2d_1.get_output(0))

    conv1_relu_1 = network.add_activation(conv1_bn2d_1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv1.3.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv1_conv2d_2 = network.add_convolution_nd(conv1_relu_1.get_output(0), 64, [3, 3], trt.Weights(w), trt.Weights(b))
    conv1_conv2d_2.padding_nd = [1, 1]
    conv1_conv2d_2.stride_nd = [1, 1]

    conv1_bn2d_2 = add_batch_norm_2d(network, para, "conv1.4", conv1_conv2d_2.get_output(0))

    conv1_relu_2 = network.add_activation(conv1_bn2d_2.get_output(0), trt.ActivationType.RELU)

    conv1_pool2d = network.add_pooling_nd(conv1_relu_2.get_output(0), trt.PoolingType.MAX, [2, 2])
    conv1_pool2d.stride_nd = [2, 2]

    # 第二层，2个卷积层和一个最大池化层
    w = np.ascontiguousarray(para["conv2.0.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv2_conv2d_1 = network.add_convolution_nd(conv1_pool2d.get_output(0), 128, [3, 3], trt.Weights(w), trt.Weights(b))
    conv2_conv2d_1.padding_nd = [1, 1]
    conv2_conv2d_1.stride_nd = [1, 1]

    conv2_bn2d_1 = add_batch_norm_2d(network, para, "conv2.1", conv2_conv2d_1.get_output(0))

    conv2_relu_1 = network.add_activation(conv2_bn2d_1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv2.3.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv2_conv2d_2 = network.add_convolution_nd(conv2_relu_1.get_output(0), 128, [3, 3], trt.Weights(w), trt.Weights(b))
    conv2_conv2d_2.padding_nd = [1, 1]
    conv2_conv2d_2.stride_nd = [1, 1]

    conv2_bn2d_2 = add_batch_norm_2d(network, para, "conv2.4", conv2_conv2d_2.get_output(0))

    conv2_relu_2 = network.add_activation(conv2_bn2d_2.get_output(0), trt.ActivationType.RELU)

    conv2_pool2d = network.add_pooling_nd(conv2_relu_2.get_output(0), trt.PoolingType.MAX, [2, 2])
    conv2_pool2d.stride_nd = [2, 2]

    # 第三层，3个卷积层和一个最大池化层
    w = np.ascontiguousarray(para["conv3.0.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv3_conv2d_1 = network.add_convolution_nd(conv2_pool2d.get_output(0), 256, [3, 3], trt.Weights(w), trt.Weights(b))
    conv3_conv2d_1.padding_nd = [1, 1]
    conv3_conv2d_1.stride_nd = [1, 1]

    conv3_bn2d_1 = add_batch_norm_2d(network, para, "conv3.1", conv3_conv2d_1.get_output(0))

    conv3_relu_1 = network.add_activation(conv3_bn2d_1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv3.3.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv3_conv2d_2 = network.add_convolution_nd(conv3_relu_1.get_output(0), 256, [3, 3], trt.Weights(w), trt.Weights(b))
    conv3_conv2d_2.padding_nd = [1, 1]
    conv3_conv2d_2.stride_nd = [1, 1]

    conv3_bn2d_2 = add_batch_norm_2d(network, para, "conv3.4", conv3_conv2d_2.get_output(0))

    conv3_relu_2 = network.add_activation(conv3_bn2d_2.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv3.6.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv3_conv2d_3 = network.add_convolution_nd(conv3_relu_2.get_output(0), 256, [3, 3], trt.Weights(w), trt.Weights(b))
    conv3_conv2d_3.padding_nd = [1, 1]
    conv3_conv2d_3.stride_nd = [1, 1]

    conv3_bn2d_3 = add_batch_norm_2d(network, para, "conv3.7", conv3_conv2d_3.get_output(0))

    conv3_relu_3 = network.add_activation(conv3_bn2d_3.get_output(0), trt.ActivationType.RELU)

    conv3_pool2d = network.add_pooling_nd(conv3_relu_3.get_output(0), trt.PoolingType.MAX, [2, 2])
    conv3_pool2d.stride_nd = [2, 2]

    # 第四层，3个卷积层和1个最大池化层
    w = np.ascontiguousarray(para["conv4.0.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv4_conv2d_1 = network.add_convolution_nd(conv3_pool2d.get_output(0), 512, [3, 3], trt.Weights(w), trt.Weights(b))
    conv4_conv2d_1.padding_nd = [1, 1]
    conv4_conv2d_1.stride_nd = [1, 1]

    conv4_bn2d_1 = add_batch_norm_2d(network, para, "conv4.1", conv4_conv2d_1.get_output(0))

    conv4_relu_1 = network.add_activation(conv4_bn2d_1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv4.3.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv4_conv2d_2 = network.add_convolution_nd(conv4_relu_1.get_output(0), 512, [3, 3], trt.Weights(w), trt.Weights(b))
    conv4_conv2d_2.padding_nd = [1, 1]
    conv4_conv2d_2.stride_nd = [1, 1]

    conv4_bn2d_2 = add_batch_norm_2d(network, para, "conv4.4", conv4_conv2d_2.get_output(0))

    conv4_relu_2 = network.add_activation(conv4_bn2d_2.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv4.6.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv4_conv2d_3 = network.add_convolution_nd(conv4_relu_2.get_output(0), 512, [3, 3], trt.Weights(w), trt.Weights(b))
    conv4_conv2d_3.padding_nd = [1, 1]
    conv4_conv2d_3.stride_nd = [1, 1]

    conv4_bn2d_3 = add_batch_norm_2d(network, para, "conv4.7", conv4_conv2d_3.get_output(0))

    conv4_relu_3 = network.add_activation(conv4_bn2d_3.get_output(0), trt.ActivationType.RELU)

    conv4_pool2d = network.add_pooling_nd(conv4_relu_3.get_output(0), trt.PoolingType.MAX, [2, 2])
    conv4_pool2d.stride_nd = [2, 2]

    # 第五层，3个卷积层和1个最大池化层
    w = np.ascontiguousarray(para["conv5.0.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv5_conv2d_1 = network.add_convolution_nd(conv4_pool2d.get_output(0), 512, [3, 3], trt.Weights(w), trt.Weights(b))
    conv5_conv2d_1.padding_nd = [1, 1]
    conv5_conv2d_1.stride_nd = [1, 1]

    conv5_bn2d_1 = add_batch_norm_2d(network, para, "conv5.1", conv5_conv2d_1.get_output(0))

    conv5_relu_1 = network.add_activation(conv5_bn2d_1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv5.3.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv5_conv2d_2 = network.add_convolution_nd(conv5_relu_1.get_output(0), 512, [3, 3], trt.Weights(w), trt.Weights(b))
    conv5_conv2d_2.padding_nd = [1, 1]
    conv5_conv2d_2.stride_nd = [1, 1]

    conv5_bn2d_2 = add_batch_norm_2d(network, para, "conv5.4", conv5_conv2d_2.get_output(0))

    conv5_relu_2 = network.add_activation(conv5_bn2d_2.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["conv5.6.weight"])
    b = np.zeros(w.shape[0], dtype=np.float32)
    conv5_conv2d_3 = network.add_convolution_nd(conv5_relu_2.get_output(0), 512, [3, 3], trt.Weights(w), trt.Weights(b))
    conv5_conv2d_3.padding_nd = [1, 1]
    conv5_conv2d_3.stride_nd = [1, 1]

    conv5_bn2d_3 = add_batch_norm_2d(network, para, "conv5.7", conv5_conv2d_3.get_output(0))

    conv5_relu_3 = network.add_activation(conv5_bn2d_3.get_output(0), trt.ActivationType.RELU)

    conv5_pool2d = network.add_pooling_nd(conv5_relu_3.get_output(0), trt.PoolingType.MAX, [2, 2])
    conv5_pool2d.stride_nd = [2, 2]

    # reshape
    reshape_layer = network.add_shuffle(conv5_pool2d.get_output(0))
    reshape_layer.reshape_dims = (-1, 512 * 7 * 7)

    # 全连接层
    w = np.ascontiguousarray(para["fc.0.weight"].transpose())
    b = np.ascontiguousarray(para["fc.0.bias"].reshape(1, -1))
    fc_1_w = network.add_constant(w.shape, trt.Weights(w))
    fc_1_multiply = network.add_matrix_multiply(reshape_layer.get_output(0), trt.MatrixOperation.NONE,
                                                fc_1_w.get_output(0), trt.MatrixOperation.NONE)
    fc_1_b = network.add_constant(b.shape, trt.Weights(b))
    fc_1 = network.add_elementwise(fc_1_multiply.get_output(0), fc_1_b.get_output(0), trt.ElementWiseOperation.SUM)
    fc_1_relu = network.add_activation(fc_1.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["fc.3.weight"].transpose())
    b = np.ascontiguousarray(para["fc.3.bias"].reshape(1, -1))
    fc_2_w = network.add_constant(w.shape, trt.Weights(w))
    fc_2_multiply = network.add_matrix_multiply(fc_1_relu.get_output(0), trt.MatrixOperation.NONE,
                                                fc_2_w.get_output(0), trt.MatrixOperation.NONE)
    fc_2_b = network.add_constant(b.shape, trt.Weights(b))
    fc_2 = network.add_elementwise(fc_2_multiply.get_output(0), fc_2_b.get_output(0), trt.ElementWiseOperation.SUM)
    fc_2_relu = network.add_activation(fc_2.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["fc.6.weight"].transpose())
    b = np.ascontiguousarray(para["fc.6.bias"].reshape(1, -1))
    fc_3_w = network.add_constant(w.shape, trt.Weights(w))
    fc_3_multiply = network.add_matrix_multiply(fc_2_relu.get_output(0), trt.MatrixOperation.NONE,
                                                fc_3_w.get_output(0), trt.MatrixOperation.NONE)
    fc_3_b = network.add_constant(b.shape, trt.Weights(b))
    fc_3 = network.add_elementwise(fc_3_multiply.get_output(0), fc_3_b.get_output(0), trt.ElementWiseOperation.SUM)

    fc_3.get_output(0).name = OUTPUT_NAME
    network.mark_output(fc_3.get_output(0))


def get_engine():
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
                                                             (5, 3, input_size[0], input_size[1]), cache_file)

        build_network(network, profile, config)

        engine_string = builder.build_serialized_network(network, config)
        if engine_string is None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trt_file, "wb") as f:
            f.write(engine_string)
            print("Succeeded saving .plan file!")

    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)

    return engine


def image_preprocess(np_img):
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)  # bgr to rgb
    # resize
    img = cv2.resize(img, (int(input_size[1] * 1.143), int(input_size[0] * 1.143)), interpolation=cv2.INTER_LINEAR)
    # crop
    crop_top = (img.shape[0] - input_size[0]) // 2
    crop_left = (img.shape[1] - input_size[1]) // 2
    img = img[crop_top:crop_top + input_size[0], crop_left:crop_left + input_size[1], :]
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data = img.astype(np.float32)
    data = (data / 255. - np.array(mean)) / np.array(std)
    # transpose
    data = data.transpose((2, 0, 1)).astype(np.float32)  # HWC to CHW

    return data


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

    outs = buffer_h[-1].reshape(-1)
    predict = np.exp(outs)
    predict = predict / np.sum(predict)
    cls = int(np.argmax(predict))
    score = predict[cls]

    return index2class_name[cls], score


if __name__ == '__main__':
    engine = get_engine()

    n_io = engine.num_bindings
    l_tensor_name = [engine.get_binding_name(i) for i in range(n_io)]
    n_input = np.sum([engine.binding_is_input(i) for i in range(n_io)])

    context = engine.create_execution_context()
    context.set_binding_shape(0, [1, 3, input_size[0], input_size[1]])
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

            start = time.time()
            input_data = image_preprocess(image)
            input_data = np.expand_dims(input_data, axis=0)  # add batch size dimension

            cate, prob = inference_one(input_data, context, buffer_h, buffer_d)
            print("Image name: %20s, Classify: %10s, prob: %.2f" % (image_name, cate, prob))
            end = time.time()

            total_cost += end - start
            img_count += 1

    print("Total image num is: %d, inference total cost is: %.3f, average cost is: %.3f" % (
        img_count, total_cost, total_cost / img_count))

    for bd in buffer_d:
        cudart.cudaFree(bd)
