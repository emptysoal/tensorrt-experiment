# -*- coding:utf-8 -*-

"""
    PyTorch 模型转 npz 文件
"""

import torch
import numpy as np

from vgg import VGG16

classes_num = 5
input_size = (224, 224)  # (rows, cols)
pth_file = "../../../PyTorch/model/best.pth"
para_file = "./para.npz"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    print('cuda device count: ', torch.cuda.device_count())

    net = VGG16(num_classes=classes_num)
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

"""
Key:  conv1.0.weight Value:  torch.Size([64, 3, 3, 3])  To numpy shape:  (64, 3, 3, 3)
Key:  conv1.1.weight Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.1.bias Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.1.running_mean Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.1.running_var Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.1.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv1.3.weight Value:  torch.Size([64, 64, 3, 3])  To numpy shape:  (64, 64, 3, 3)
Key:  conv1.4.weight Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.4.bias Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.4.running_mean Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.4.running_var Value:  torch.Size([64])  To numpy shape:  (64,)
Key:  conv1.4.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv2.0.weight Value:  torch.Size([128, 64, 3, 3])  To numpy shape:  (128, 64, 3, 3)
Key:  conv2.1.weight Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.1.bias Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.1.running_mean Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.1.running_var Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.1.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv2.3.weight Value:  torch.Size([128, 128, 3, 3])  To numpy shape:  (128, 128, 3, 3)
Key:  conv2.4.weight Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.4.bias Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.4.running_mean Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.4.running_var Value:  torch.Size([128])  To numpy shape:  (128,)
Key:  conv2.4.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv3.0.weight Value:  torch.Size([256, 128, 3, 3])  To numpy shape:  (256, 128, 3, 3)
Key:  conv3.1.weight Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.1.bias Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.1.running_mean Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.1.running_var Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.1.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv3.3.weight Value:  torch.Size([256, 256, 3, 3])  To numpy shape:  (256, 256, 3, 3)
Key:  conv3.4.weight Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.4.bias Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.4.running_mean Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.4.running_var Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.4.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv3.6.weight Value:  torch.Size([256, 256, 3, 3])  To numpy shape:  (256, 256, 3, 3)
Key:  conv3.7.weight Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.7.bias Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.7.running_mean Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.7.running_var Value:  torch.Size([256])  To numpy shape:  (256,)
Key:  conv3.7.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv4.0.weight Value:  torch.Size([512, 256, 3, 3])  To numpy shape:  (512, 256, 3, 3)
Key:  conv4.1.weight Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.1.bias Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.1.running_mean Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.1.running_var Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.1.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv4.3.weight Value:  torch.Size([512, 512, 3, 3])  To numpy shape:  (512, 512, 3, 3)
Key:  conv4.4.weight Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.4.bias Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.4.running_mean Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.4.running_var Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.4.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv4.6.weight Value:  torch.Size([512, 512, 3, 3])  To numpy shape:  (512, 512, 3, 3)
Key:  conv4.7.weight Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.7.bias Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.7.running_mean Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.7.running_var Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv4.7.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv5.0.weight Value:  torch.Size([512, 512, 3, 3])  To numpy shape:  (512, 512, 3, 3)
Key:  conv5.1.weight Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.1.bias Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.1.running_mean Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.1.running_var Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.1.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv5.3.weight Value:  torch.Size([512, 512, 3, 3])  To numpy shape:  (512, 512, 3, 3)
Key:  conv5.4.weight Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.4.bias Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.4.running_mean Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.4.running_var Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.4.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  conv5.6.weight Value:  torch.Size([512, 512, 3, 3])  To numpy shape:  (512, 512, 3, 3)
Key:  conv5.7.weight Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.7.bias Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.7.running_mean Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.7.running_var Value:  torch.Size([512])  To numpy shape:  (512,)
Key:  conv5.7.num_batches_tracked Value:  torch.Size([])  To numpy shape:  ()
Key:  fc.0.weight Value:  torch.Size([4096, 25088])  To numpy shape:  (4096, 25088)
Key:  fc.0.bias Value:  torch.Size([4096])  To numpy shape:  (4096,)
Key:  fc.3.weight Value:  torch.Size([4096, 4096])  To numpy shape:  (4096, 4096)
Key:  fc.3.bias Value:  torch.Size([4096])  To numpy shape:  (4096,)
Key:  fc.6.weight Value:  torch.Size([5, 4096])  To numpy shape:  (5, 4096)
Key:  fc.6.bias Value:  torch.Size([5])  To numpy shape:  (5,)
"""
