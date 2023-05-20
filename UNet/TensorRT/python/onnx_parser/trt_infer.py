# -*- coding:utf-8 -*-

"""
    onnx 模型转 tensorrt 模型，并使用 tensorrt runtime 推理
    与 trt_infer 的不同：
        trt_infer  将 topk 操作写在后处理中（通过 np.argmax 实现的），engine 的输出尺寸为 1 * cls * height * width
        trt_infer2 将 topk 操作写在 network 构建过程最后（通过 network.add_topk 实现的），engine 的输出尺寸为 1 * 1 * height * width
"""

import os
import time

import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart

import calibrator
from mask2color import to_color

classes_num = 32
input_height = 448
input_width = 448
onnx_file = "./model.onnx"
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

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnx_file):
            print("Failed finding ONNX file!")
            return
        print("Succeeded finding ONNX file!")
        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
            print("Succeeded parsing .onnx file!")

        input_tensor = network.get_input(0)
        profile.set_shape(input_tensor.name, [1, 3, input_height, input_width], [4, 3, input_height, input_width],
                          [8, 3, input_height, input_width])
        config.add_optimization_profile(profile)

        # unmark output(original output is conv layer)
        output_tensor = network.get_output(0)
        network.unmark_output(output_tensor)
        # add topk layer
        top1_layer = network.add_topk(output_tensor, trt.TopKOperation.MAX, 1, 1 << 1)
        top1_layer.get_output(1).name = "mask"
        network.mark_output(top1_layer.get_output(1))

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

    for b in buffer_d:
        cudart.cudaFree(b)
