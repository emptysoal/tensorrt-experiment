# -*- coding:utf-8 -*-

import os

from PIL import Image
from torch.utils.data import Dataset

class_name2index = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}
index2class_name = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}


class FlowerDataset(Dataset):
    def __init__(self, images_dir: str, transform=None):
        self.images_dir = images_dir  # 训练或验证数据集目录
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for per_class in os.listdir(self.images_dir):
            per_class_path = os.path.join(self.images_dir, per_class)
            label = class_name2index[per_class]

            for per_image_name in os.listdir(per_class_path):
                per_image_path = os.path.join(per_class_path, per_image_name)
                self.image_paths.append(per_image_path)
                self.labels.append(label)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        label = self.labels[index]
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
