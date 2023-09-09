# Comparative experiment of inference speed for different TensorRT api

## Introduction

- Based on TensorRT **8.2.4** ，see the environment section below for details

- **Target:**

  **Compare inference speed for pytorch、onnx runtime、tensorrt C++(include onnxparser、definition api)、tensorrt python(include onnxparser、definition api)**

- Process：

1. Using PyTorch to implement or collect some classical CNN network, training to get `.pth` model file; 
2. Python api with tensorrt and cuda :
   - OnnxParser build network: model.pth -> model.onnx -> model.plan；
   - Definition api build network layer by layer: model.pth -> model.npz-> model.plan；
3. C++ api with tensorrt and cuda :
   - OnnxParser build network: model.pth -> model.onnx -> model.plan；
   - Definition api build network layer by layer: model.pth -> model.wts-> model.plan；
4. Compare time cost for different inference api;

## Result

- TensorRT FP32

|            | PyTorch |  ONNX  | Python trt onnxparser | Python trt api | C++ trt onnxparser | C++ trt api |
| ---------- | :-----: | :----: | :-------------------: | :------------: | :----------------: | :---------: |
| VGG16      |  93 ms  | 74 ms  |         9 ms          |      9 ms      |        5 ms        |    5 ms     |
| ResNet50   |  96 ms  | 75 ms  |         9 ms          |      9 ms      |        5 ms        |    5 ms     |
| UNet       | 181 ms  | 152 ms |         26 ms         |     26 ms      |       27 ms        |    26 ms    |
| Deeplabv3+ | 208 ms  | 158 ms |         28 ms         |     30 ms      |       26 ms        |    25 ms    |

- TensorRT **INT8**

|            | Python trt onnxparser | Python trt api | C++ trt onnxparser | C++ trt api |
| ---------- | :-------------------: | :------------: | :----------------: | :---------: |
| ResNet50   |         6 ms          |      6 ms      |        3 ms        |    2 ms     |
| Deeplabv3+ |         15 ms         |     15 ms      |       11 ms        |    10 ms    |

- Add: precision comparison before and after  int8  quantization

|                       | before quantization | after quantization |
| --------------------- | :-----------------: | :----------------: |
| ResNet50（precision） |       95.08%        |       95.08%       |
| Deeplabv3+（mIOU）    |       61.99%        |       60.96%       |

Note：backbone of Deeplabv3+  is resnet50

## Reference

```bash
https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/  # TensorRT official document(C++ api)
https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/  # TensorRT official document(python api)
https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook
https://github.com/wang-xinyu/tensorrtx
```

## File descripition

```bash
project dir
	├── flower_classify_dataset  # Dataset used for classification, 5 kind of flowers, link below 
	│   ├── train
	│   ├── val
	│   └── test
    ├── Camvid_segment_dataset  # Dataset used for segmentation，link below 
    │   ├── images  # original images
    |   │   ├── train
    |   │   ├── val
    |   │   └── test
    │   ├── labels  # mask labels，Consists of the category index 
    │   ├── train.lst  # format: train_image_path'\t'train_label_path
    │   ├── val.lst  # format: valid_image_path'\t'valid_label_path
	│   └── labels.txt  # category index and it's color
	├── VGG16
	│   ├── PyTorch
    │   │   ├── dataset.py
    │   │   ├── model  # auto generated after running train.py, '.pth' type model file will be here
    │   │   ├── pytorch_inference.py  # use pytorch api inference
    │   │   ├── train.py
    │   │   └── vgg.py  # pytorch network file
    │   └── TensorRT
    │       ├── C++
    │       │   ├── api_model
    │       │   │   ├── calibrator.cpp  # for int8 quantization
    │       │   │   ├── calibrator.h
    │       │   │   ├── Makefile
    │       │   │   ├── pth2wts.py  # '.pth' type model file to '.wts' type file
    │       │   │   ├── public.h
    │       │   │   ├── trt_infer.cpp  # generate tensorrt plan file, use tensorrt c++ api inference
    │       │   │   └── vgg.py  # it's the same file as above and below
    │       │   └── onnx_parser
    │       │       ├── calibrator.cpp
    │       │       ├── calibrator.h
    │       │       ├── Makefile
    │       │       ├── onnx_infer.py  # '.pth' type file to '.onnx' type file, and onnx runtime inference
    │       │       ├── public.h
    │       │       ├── trt_infer.cpp  # onnx to tensorrt plan file, and tensorrt c++ api inference
    │       │       └── vgg.py
    │       └── python
    │           ├── api_model
    │           │   ├── calibrator.py  # for int8 quantization
    │           │   ├── pth2npz.py  # '.pth' type model file to '.npz' type file
    │           │   ├── trt_inference.py  # generate tensorrt plan file, use tensorrt python api inference
    │           │   └── vgg.py
    │           └── onnx_parser
    │               ├── calibrator.py
    │               ├── onnx_infer.py  # '.pth' type file to '.onnx' type file, and onnx runtime inference
    │               ├── trt_infer.py  # onnx to tensorrt plan file, and tensorrt python api inference
    │               └── vgg.py
    ├── ResNet50  # The file structure is basically the same as VGG16
	│   ├── PyTorch
	│   └── TensorRT
	│       ├── C++
	│       │   ├── api_model
	│       │   └── onnx_parser
	│       └── python
	│           ├── api_model
	│           └── onnx_parser
	├── UNet  # The file structure is basically the same as VGG16
	│   ├── PyTorch
	│   └── TensorRT
	│       ├── C++
	│       │   ├── api_model
	│       │   └── onnx_parser
	│       └── python
	│           ├── api_model
	│           └── onnx_parser
	└── Deeplabv3+  # The file structure is basically the same as VGG16
	    ├── PyTorch
	    └── TensorRT
	        ├── C++
	        │   ├── api_model
	        │   └── onnx_parser
	        └── python
	            ├── api_model
	            └── onnx_parser
```

- For details on the operation of each subproject, please read README under the specific subproject directory 
- datasets link：[dataset](https://pan.baidu.com/s/1n-kluXn3XdrHuaZUjSa4fQ)  extract code：z3qp

## Environment

### Base environment

- Ubuntu 16.04
- GPU：GeForce RTX 2080 Ti
- CUDA 11.2
- docker，nvidia-docker

### Pull base image

```bash
docker pull nvcr.io/nvidia/tensorrt:22.04-py3
```

- The library versions in the image are as follows:

| CUDA   | cuDNN    | TensorRT | python |
| ------ | -------- | -------- | ------ |
| 11.6.2 | 8.4.0.27 | 8.2.4.2  | 3.8.10 |

### Install other libraries

1. build docker container

   ```bash
   docker run -it --gpus device=0 --shm-size 32G -v /home:/workspace nvcr.io/nvidia/tensorrt:22.04-py3 bash
   ```

   -v /home:/workspace Mounts the /home directory of the host to the container to facilitate file interaction. You can also select other directories 

2. install OpenCV-4.5.0

   - OpenCV-4.5.0 source link is as follows, download the zip package, decompress it and put it in the host `/home` directory, that is, the container `/workspace` directory 

   ```bash
   https://github.com/opencv/opencv
   ```

   - The following operations are in the container

   ```bash
   # Install dependency
   apt install build-essential
   apt install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
   apt install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
   # start to install OpenCV
   cd /workspace/opencv-4.5.0
   mkdir build
   cd build
   cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=Release -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_ENABLE_NONFREE=True ..
   make -j6
   make install
   ```

3. install PyTorch

   - download  `torch-1.12.0`

   ```bash
   # open link: https://download.pytorch.org/whl/torch/  # pytorch official website
   # find torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
   # download and place in /workspace directory
   # run:
   pip install torch-1.12.0+cu116-cp38-cp38-linux_x86_64.whl
   ```

   - download  `torchvision-0.13.0`

   ```bash
   # open link https://download.pytorch.org/whl/torchvision/
   # find torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
   # download and place in /workspace directory
   # run:
   pip install torchvision-0.13.0+cu116-cp38-cp38-linux_x86_64.whl
   ```

4. install other python library

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


At this point, all programs in the project can be run.