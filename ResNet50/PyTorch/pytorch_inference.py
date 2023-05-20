# -*- coding:utf-8 -*-

import os
import time

from PIL import Image
import torch
from torchvision import transforms

# from resnet50 import ResNet50
from concise_resnet50 import ResNet50

model_file = "./model/best.pth"
input_size = 224
classes_num = 5
index2class_name = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}
infer_transform = transforms.Compose([
    transforms.Resize(int(input_size * 1.143)),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet50(classes_num=classes_num)
model.load_state_dict(torch.load(model_file, map_location='cpu'))
model.to(device)


def inference_one(data_path):
    image = Image.open(data_path)
    image_tensor = infer_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        model.eval()
        out = model(image_tensor)
        # print(out)
        # 执行softmax
        ps = torch.exp(out)
        ps = ps / torch.sum(ps)

        score, cls = ps.topk(1, dim=1)
        # print(score)

    return index2class_name[cls.item()], score.item()


if __name__ == '__main__':
    start = time.time()

    image_dir = "../../flower_classify_dataset/test"
    img_count = 0
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if image_path.endswith("jpg") or image_path.endswith("jpeg"):
            cate, prob = inference_one(image_path)
            print("Image name: %20s, Classify: %10s, prob: %.2f" % (image_name, cate, prob))
            img_count += 1

    end = time.time()
    print("Total image num is: %d, inference total cost is: %.3f, average cost is: %.3f" % (
        img_count, end - start, (end - start) / img_count))
