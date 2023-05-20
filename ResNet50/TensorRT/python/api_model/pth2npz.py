# -*- coding:utf-8 -*-

"""
    PyTorch 模型转 npz 文件
"""

import torch
import numpy as np

from concise_resnet50 import ResNet50

classes_num = 5
input_size = (224, 224)  # (rows, cols)
pth_file = "../../../PyTorch/model/best.pth"
para_file = "./para.npz"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    print('cuda device count: ', torch.cuda.device_count())

    net = ResNet50(classes_num=classes_num)
    net.load_state_dict(torch.load(pth_file, map_location="cpu"))
    net.to(device)
    net.eval()
    print('model: ', net)
    tmp = torch.ones(1, 3, input_size[0], input_size[1]).to(device)
    print('input: ', tmp)
    out = net(tmp)
    print('output:', out)

    para = {}  # save weight as npz file
    for k, v in net.state_dict().items():
        para[k] = v.cpu().numpy()
        print("Key: ", k, "Value: ", v.shape, " To numpy shape: ", v.cpu().numpy().shape)
    np.savez(para_file, **para)


if __name__ == '__main__':
    main()
