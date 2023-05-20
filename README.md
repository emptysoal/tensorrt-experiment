# TensorRT 下不同 API 推理时间的对比实验

## 项目简介

- 基于 TensorRT **8.2.4** 版本，具体环境见下面的环境构建部分

- **目标：**

  **对比 pytorch、onnx runtime、tensorrt C++（包括onnxparser、原生api）、tensorrt python（包括onnxparser、原生api）等不同框架的推理速度**

- 流程：

1. 使用 PyTorch 实现或收集了一些经典的 CNN 网络，训练得到`.pth`模型文件；
2. TensorRT 和 cuda 的 python api :
   - OnnxParser构建网络：model.pth -> model.onnx -> model.plan；
   - TensorRT API逐层构建网络：model.pth -> model.npz-> model.plan；
3. TensorRT 和 cuda 的 C++ api :
   - OnnxParser构建网络：model.pth -> model.onnx -> model.plan；
   - TensorRT API逐层构建网络：model.pth -> model.wts-> model.plan；
4. 分别对比 python 和 C++ api 下，pytorch、onnx、tensorrt 等模型的推理速度。

## 对比结果

- TensorRT FP32 精度

|            | PyTorch |  ONNX  | Python trt onnxparser | Python trt api | C++ trt onnxparser | C++ trt api |
| ---------- | :-----: | :----: | :-------------------: | :------------: | :----------------: | :---------: |
| VGG16      |  51 ms  | 37 ms  |         8 ms          |      8 ms      |        5 ms        |    5 ms     |
| ResNet50   |  53 ms  | 32 ms  |         7 ms          |      7 ms      |        4 ms        |    4 ms     |
| UNet       | 132 ms  | 129 ms |         26 ms         |     26 ms      |       26 ms        |    23 ms    |
| Deeplabv3+ | 175 ms  | 123 ms |         63 ms         |     59 ms      |       61 ms        |    57 ms    |

- TensorRT **Int8量化**

|            | Python trt onnxparser | Python trt api | C++ trt onnxparser | C++ trt api |
| ---------- | :-------------------: | :------------: | :----------------: | :---------: |
| ResNet50   |         6 ms          |      6 ms      |        3 ms        |    3 ms     |
| Deeplabv3+ |         53 ms         |     50 ms      |       52 ms        |    48 ms    |

补充：int8量化前后精度对比

|                       | before quantization | after quantization |
| --------------------- | :-----------------: | :----------------: |
| ResNet50（precision） |       95.08%        |       95.08%       |
| Deeplabv3+（mIOU）    |       61.99%        |       60.96%       |

备注：Deeplabv3+ 的 backbone 为 resnet50

## 文件说明

```bash
project dir
    ├── flower_classify_dataset  # 分类项目所用到的数据集，5种花的分类，完整数据集可通过下方链接获取
    │   ├── train
    │   ├── val
    │   └── test
    ├── Camvid_segment_dataset  # 语义分割项目所用到的数据集，完整数据集可通过下方链接获取
    │   ├── images  # 原始图像目录
    |   │   ├── train
    |   │   ├── val
    |   │   └── test
    │   ├── labels  # mask标签目录，类别index所构成
    │   ├── train.lst  # 原始图像路径\t标签路径
    │   ├── val.lst  # 原始图像路径\t标签路径
    │   └── labels.txt  # 标签index和其所对应的颜色及类别
    ├── VGG16
    │   ├── PyTorch
    │   │   ├── dataset.py
    │   │   ├── model  # 运行train.py时自动生成，pth模型会保存到其中
    │   │   ├── pytorch_inference.py  # 使用pytorch推理
    │   │   ├── train.py  # 启动训练文件
    │   │   └── vgg.py  # 模型文件
    │   └── TensorRT
    │       ├── C++
    │       │   ├── api_model
    │       │   │   ├── calibrator.cpp  # int8量化
    │       │   │   ├── calibrator.h
    │       │   │   ├── Makefile
    │       │   │   ├── pth2wts.py  # pth模型文件转wts文件
    │       │   │   ├── public.h
    │       │   │   ├── trt_infer.cpp  # 构建 tensorrt engine 及使用 engine 推理
    │       │   │   └── vgg.py
    │       │   └── onnx_parser
    │       │       ├── calibrator.cpp
    │       │       ├── calibrator.h
    │       │       ├── Makefile
    │       │       ├── onnx_infer.py  # pth模型文件转onnx文件 并使用 onnx runtime 推理
    │       │       ├── public.h
    │       │       ├── trt_infer.cpp  # onnx 转 tensorrt engine 及使用 engine 推理
    │       │       └── vgg.py
    │       └── python
    │           ├── api_model
    │           │   ├── calibrator.py  # int8量化
    │           │   ├── pth2npz.py  # pth模型文件转 numpy 文件
    │           │   ├── trt_inference.py  # 构建 tensorrt engine 及使用 engine 推理
    │           │   └── vgg.py
    │           └── onnx_parser
    │               ├── calibrator.py
    │               ├── onnx_infer.py  # pth模型文件转onnx文件 并使用 onnx runtime 推理
    │               ├── trt_infer.py  # onnx 转 tensorrt engine 及使用 engine 推理
    │               └── vgg.py
    ├── ResNet50  # 文件结构基本同VGG16
    │   ├── PyTorch
    │   └── TensorRT
    │       ├── C++
    │       │   ├── api_model
    │       │   └── onnx_parser
    │       └── python
    │           ├── api_model
    │           └── onnx_parser
    ├── UNet  # 文件结构基本同VGG16
    │   ├── PyTorch
    │   └── TensorRT
    │       ├── C++
    │       │   ├── api_model
    │       │   └── onnx_parser
    │       └── python
    │           ├── api_model
    │           └── onnx_parser
    └── Deeplabv3+  # 文件结构基本同VGG16
        ├── PyTorch
	└── TensorRT
	    ├── C++
	    │   ├── api_model
	    │   └── onnx_parser
	    └── python
	        ├── api_model
	        └── onnx_parser
```

- 具体到各工程的运行，请阅读具体工程目录下的 README
- 数据集获取链接：[dataset](https://pan.baidu.com/s/1n-kluXn3XdrHuaZUjSa4fQ)  提取码：z3qp

## 环境构建

### 宿主机基础环境

- Ubuntu 16.04
- GPU：GeForce GTX 1080Ti
- CUDA 11.2
- docker，nvidia-docker

### 基础镜像拉取

```bash
docker pull nvcr.io/nvidia/tensorrt:22.04-py3
```

- 该镜像中各种环境版本如下：

| CUDA   | cuDNN    | TensorRT | python |
| ------ | -------- | -------- | ------ |
| 11.6.2 | 8.4.0.27 | 8.2.4.2  | 3.8.10 |

### 安装其他库

1. 创建 docker 容器

   ```bash
   docker run -it --gpus device=0 --shm-size 32G -v /home:/workspace nvcr.io/nvidia/tensorrt:22.04-py3 bash
   ```

   其中`-v /home:/workspace`将宿主机的`/home`目录挂载到容器中，方便一些文件的交互，也可以选择其他目录

   - 将容器的源换成国内源

   ```bash
   cd /etc/apt
   rm sources.list
   vim sources.list
   ```

   - 将下面内容拷贝到文件sources.list

   ```bash
   deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
   ```

   - 更新源

   ```bash
   apt update
   ```

2. 安装 OpenCV-4.5.0

   - OpenCV-4.5.0源码链接如下，下载 zip 包，解压后放到宿主机`/home`目录下，即容器的`/workspace`目录下

   ```bash
   https://github.com/opencv/opencv
   ```

   - 下面操作均在容器中

   ```bash
   # 安装依赖
   apt install build-essential
   apt install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
   apt install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
   # 开始安装 OpenCV
   cd /workspace/opencv-4.5.0
   mkdir build
   cd build
   cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=Release -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_ENABLE_NONFREE=True ..
   make -j6
   make install
   ```

3. 安装PyTorch

   - 下载`torch-1.12.0`

   ```bash
   进入链接 https://download.pytorch.org/whl/torch/
   找到 torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
   下载后放到 /workspace 目录下
   pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
   ```

   - 下载`torchvision-0.13.0`

   ```bash
   进入链接 https://download.pytorch.org/whl/torchvision/
   找到 torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
   下载后放到 /workspace 目录下
   pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
   ```

4. 安装其他 python 库

   ```bash
   pip install opencv-python==3.4.16.59
   pip install opencv-python-headless==3.4.16.59
   pip install tensorboard
   pip install cuda-python
   pip install onnxruntime-gpu==1.10
   pip install scipy
   pip install matplotlib
   pip install tqdm
   ```

   下载速度慢或超时的话，在后面加上`-i https://pypi.tuna.tsinghua.edu.cn/simple/`

至此，可运行项目中全部程序。



感谢阅毕，望不吝star，以资鼓励。
