# -*- coding:utf-8 -*-

import os
import time

import cv2
import numpy as np
import torch

from modeling.deeplab import DeepLab
from mask2color import to_color

model_file = "./run/camvid/deeplab-resnet/model_best.pth.tar"
classes_num = 32
input_height = 960
input_width = 720
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DeepLab(backbone="resnet", output_stride=16, num_classes=classes_num, sync_bn=False)
ckpt = torch.load(model_file, map_location='cpu')
model.load_state_dict(ckpt["state_dict"])
model.to(device)


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
    blur_mask = cv2.medianBlur(resized_mask, 5)
    # to color
    color_img = to_color(blur_mask)
    # rgb to bgr
    bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    # save
    cv2.imwrite(file_name.split(".")[0] + "_mask.png", bgr)


def inference_one(data_input):
    image_tensor = torch.from_numpy(data_input)
    image_tensor = image_tensor.unsqueeze(0).float()  # add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        model.eval()
        out = model(image_tensor)
        mask = torch.max(out, dim=1).indices.detach().cpu().numpy()

    return mask[0]


if __name__ == '__main__':
    image_dir = "../../Camvid_segment_dataset/images/test"
    total_cost = 0
    img_count = 0
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if image_path.endswith("jpg") or image_path.endswith("jpeg"):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # read image
            origin_height, origin_width = image.shape[:2]  # get origin resolution

            start = time.time()
            input_data = image_preprocess(image)  # image preprocess
            mask = inference_one(input_data)
            end = time.time()
            image_postprocess(mask, origin_height, origin_width, image_name)  # for visualization, not necessary

            total_cost += end - start
            img_count += 1

    print("Total image num is: %d, inference total cost is: %.3f, average cost is: %.3f" % (
        img_count, total_cost, total_cost / img_count))
