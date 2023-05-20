# -*- coding:utf-8 -*-

"""
    PyTorch 模型转 wts 文件
"""

import struct
import torch
from unet.unet_model import UNet

classes_num = 32
input_height = 448
input_width = 448
pth_file = "../../../PyTorch/model/best.pth"
wts_file = "./para.wts"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    print('cuda device count: ', torch.cuda.device_count())

    net = UNet(n_channels=3, n_classes=classes_num)
    net.load_state_dict(torch.load(pth_file, map_location="cpu"))
    net.to(device)
    net.eval()
    print('model: ', net)
    tmp = torch.ones(1, 3, input_height, input_width).to(device)
    print('input: ', tmp)
    out = net(tmp)
    print('output:', out)

    f = open(wts_file, "w")
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k, v in net.state_dict().items():
        print("Key: ", k, " Value: ", v.shape, " To numpy shape: ", v.cpu().numpy().shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")


if __name__ == '__main__':
    main()
