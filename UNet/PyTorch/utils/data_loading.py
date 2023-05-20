import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        # 获取文件名，不包括拓展名
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if
                    os.path.isfile(os.path.join(images_dir, file)) and not file.startswith(".")]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        new_w, new_h = int(w * scale), int(h * scale)
        assert new_w > 0 and new_h > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            return img

        img = img.transpose((2, 0, 1))
        img = img / 255.0
        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = os.path.join(self.images_dir, name + ".jpg")
        mask_path = os.path.join(self.mask_dir, name + ".png")
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


if __name__ == '__main__':
    camvid_images_dir = "../../Camvid_segment_dataset/images/train"
    camvid_labels_dir = "../../Camvid_segment_dataset/labels/train"
    dataset = BasicDataset(camvid_images_dir, camvid_labels_dir)
    for i, item in enumerate(dataset):
        print(item["image"].size(), item["mask"].size())
