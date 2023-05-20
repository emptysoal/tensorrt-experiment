- 训练

```bash
cd PyTorch
python train.py --epochs 500 --learning-rate 0.0005 --pre-model-dir ./pretrained/resnet50-0676ba61.pth
```

- PyTorch推理

```bash
cd PyTorch
python pytorch_inference.py
```

- TensorRT  python onnxparser模型转换

```bash
cd TensorRT/python/onnx_parser
python onnx_infer.py  # pth模型文件转onnx文件 并使用 onnx runtime 推理
python trt_infer.py   # onnx 转 tensorrt engine 及使用 engine 推理
```

- TensorRT  python api 搭建网络

```bash
cd TensorRT/python/api_model
python pth2npz.py     # pth模型文件转npz文件
python trt_infer.py   # 构建 tensorrt engine 及使用 engine 推理
```

- TensorRT  C++ onnxparser模型转换

```bash
cd TensorRT/C++/onnx_parser
python onnx_infer.py  # pth模型文件转onnx文件 并使用 onnx runtime 推理
make
./trt_infer           # onnx 转 tensorrt engine 及使用 engine 推理
```

- TensorRT  C++ api 搭建网络

```bash
cd TensorRT/C++/api_model
python pth2wts.py     # pth模型文件转wts文件
make
./trt_infer           # 构建 tensorrt engine 及使用 engine 推理
```

