# -*- coding:utf-8 -*-

"""
    PyTorch 模型转 onnx 模型，并使用 onnx runtime 模型推理
"""

import os
import time

import cv2
import numpy as np
import torch
import onnxruntime

from unet.unet_model import UNet
from mask2color import to_color

classes_num = 32
input_height = 448  # 转换时去掉了上采样时的pad操作，所以输入尺寸必须是32的整数倍（最大下采样为32倍）
input_width = 448
pth_file = "../../../PyTorch/model/best.pth"
onnx_file = "./model.onnx"
data_path = "../../../../Camvid_segment_dataset"
test_data_path = data_path + "/images/test"  # 用于推理

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pth2onnx():
    """
        PyTorch模型转onnx模型
    """
    model = UNet(n_channels=3, n_classes=classes_num)
    model.load_state_dict(torch.load(pth_file, map_location="cpu"))
    model.to(device)

    torch.onnx.export(
        model,
        torch.randn((1, 3, input_height, input_width), device=device),
        onnx_file,
        input_names=["data"],
        output_names=["prob"],
        do_constant_folding=True,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=12,
        dynamic_axes={"data": {0: "nBatchSize"}, "prob": {0: "nBatchSize"}}
    )
    print("Succeeded converting model into ONNX!")


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


def inference_one(data_input, onnx_session):
    inputs = {onnx_session.get_inputs()[0].name: data_input}
    outs = onnx_session.run(None, inputs)[0]

    mask = np.argmax(outs, axis=1)

    return mask[0]


if __name__ == '__main__':
    if not os.path.exists(onnx_file):  # 不存在onnx模型就使用pth模型导出
        pth2onnx()

    # onnx inference
    session = onnxruntime.InferenceSession(
        onnx_file,
        providers=[
            # 'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            # 'CPUExecutionProvider'
        ]
    )

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

            output = inference_one(input_data, session)
            end = time.time()

            image_postprocess(output, origin_height, origin_width, image_name)  # for visualization, not necessary

            total_cost += end - start
            img_count += 1

    print("Total image num is: %d, inference total cost is: %.3f, average cost is: %.3f" % (
        img_count, total_cost, total_cost / img_count))
