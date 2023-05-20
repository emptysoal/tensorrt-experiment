CUDA_PATH       = /usr/local/cuda
TRT_INC_PATH    = /usr/include/x86_64-linux-gnu
TRT_LIB_PATH    = /usr/lib/x86_64-linux-gnu
OPENCV_INC_PATH = /usr/local/include/opencv4
OPENCV_LIB_PATH = /usr/local/lib
INCLUDE         = -I$(CUDA_PATH)/include -I$(TRT_INC_PATH) -I$(OPENCV_INC_PATH)
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart -L$(TRT_LIB_PATH) -lnvinfer -lnvonnxparser
LDFLAG         += -L$(OPENCV_LIB_PATH) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_dnn

CC = g++

all: trt_infer

trt_infer: trt_infer.cpp calibrator.cpp
	$(CC) calibrator.cpp trt_infer.cpp -o trt_infer $(INCLUDE) $(LDFLAG) -lz

clean:
	rm -rf ./trt_infer ./*.plan ./*.cache
