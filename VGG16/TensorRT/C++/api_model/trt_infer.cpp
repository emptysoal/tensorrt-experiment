#include "public.h"
#include "calibrator.h"

using namespace nvinfer1;

const char *      inputName = "data";
const char *      outputName = "prob";
const int         inputHeight = 224;
const int         inputWidth = 224;
const std::string wtsFile = "./para.wts";
const std::string trtFile = "./model.plan";
const std::string dataPath = "../../../../flower_classify_dataset";
const std::string valDataPath = dataPath + "/val";
const std::string testDataPath = dataPath + "/test";  // 用于推理
// const std::string testImagePath = dataPath + "/test/roses_01.jpg";  // 用于推理
const int         classesNum = 5;
static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool        bFP16Mode = false;
// for INT8 mode
const bool        bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = dataPath + "/int8";  // 用于 int8 量化


// refer pytorch training
void imagePreProcess(cv::Mat& img, float* inputData)
{
    /*
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inp_image = ((resized_img/255. - mean) / std).astype(np.float32)
    */

    // resize
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(int(inputWidth * 1.143), int(inputHeight * 1.143)));
    // center crop
    int crop_top = (resize_img.rows - inputHeight) / 2;
    int crop_left = (resize_img.cols - inputWidth) / 2;
    cv::Mat cropped_img(resize_img, cv::Range(crop_top, crop_top + inputHeight), cv::Range(crop_left, crop_left + inputWidth));

    cv::Mat pr_img;
    pr_img = cropped_img.clone();

    int i = 0;
    for (int row = 0; row < inputHeight; row++)
    {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < inputWidth; col++)
        {
            inputData[i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;  // R-0.485
            inputData[i + inputHeight * inputWidth] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
            inputData[i + 2 * inputHeight * inputWidth] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
            uc_pixel += 3;
            i++;
        }
    }
}

void imagePostProcess(std::string file_name, float* outputs)
{
    // softmax
    double sum = 0;
    for (int i = 0; i < classesNum; i++) sum += exp(double(outputs[i]));
    std::vector<double> softmaxOutputs(classesNum, 0);  // prob value
    for (int i = 0; i < classesNum; i++) softmaxOutputs[i] = exp(double(outputs[i])) / sum;
    int maxValIdx = 0;  // 先把第一个当作最大值，然后后面的依次与它比较
    for (int i = 1; i < classesNum; i++)
    {
        if (softmaxOutputs[i] > softmaxOutputs[maxValIdx]) maxValIdx = i;
    }

    std::map<int, std::string> index2classMap;
    index2classMap.insert(std::pair<int, std::string>(0, "daisy"));
    index2classMap.insert(std::pair<int, std::string>(1, "dandelion"));
    index2classMap.insert(std::pair<int, std::string>(2, "roses"));
    index2classMap.insert(std::pair<int, std::string>(3, "sunflowers"));
    index2classMap.insert(std::pair<int, std::string>(4, "tulips"));
    std::cout << "Image name is: " << file_name;
    std::cout << " category is : " << index2classMap[maxValIdx];
    std::cout << ", prob is : ";
    std::cout << std::fixed << std::setprecision(2) << softmaxOutputs[maxValIdx] * 100 << "%." << std::endl;
}


void inference_one(IExecutionContext* context, float* inputData, float* outputData, std::vector<int>& vTensorSize)
{
    std::vector<void *> vBufferD (2, nullptr);
    for (int i = 0; i < 2; i++)
    {
        CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }
    CHECK(cudaMemcpy(vBufferD[0], (void *)inputData, vTensorSize[0], cudaMemcpyHostToDevice));

    context->executeV2(vBufferD.data());

    CHECK(cudaMemcpy((void *)outputData, vBufferD[1], vTensorSize[1], cudaMemcpyDeviceToHost));

    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i]));
    }
}


IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


void buildNetwork(INetworkDefinition* network, IOptimizationProfile* profile, IBuilderConfig* config)
{
    ITensor* inputTensor = network->addInput(inputName, DataType::kFLOAT, Dims32 {4, {-1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 3, inputHeight, inputWidth}});
    config->addOptimizationProfile(profile);

    std::map<std::string, Weights> weightMap = loadWeights(wtsFile);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // 第一层，2个卷积层和一个最大池化层
    IConvolutionLayer* conv1_conv2d_1 = network->addConvolutionNd(*inputTensor, 64, DimsHW{3, 3}, weightMap["conv1.0.weight"], emptywts);
    conv1_conv2d_1->setPaddingNd(DimsHW{1, 1});
    conv1_conv2d_1->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv1_bn2d_1 = addBatchNorm2d(network, weightMap, *conv1_conv2d_1->getOutput(0), "conv1.1", 1e-5);

    IActivationLayer* conv1_relu_1 = network->addActivation(*conv1_bn2d_1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv1_conv2d_2 = network->addConvolutionNd(*conv1_relu_1->getOutput(0), 64, DimsHW{3, 3}, weightMap["conv1.3.weight"], emptywts);
    conv1_conv2d_2->setPaddingNd(DimsHW{1, 1});
    conv1_conv2d_2->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv1_bn2d_2 = addBatchNorm2d(network, weightMap, *conv1_conv2d_2->getOutput(0), "conv1.4", 1e-5);

    IActivationLayer* conv1_relu_2 = network->addActivation(*conv1_bn2d_2->getOutput(0), ActivationType::kRELU);

    IPoolingLayer* conv1_pool2d = network->addPoolingNd(*conv1_relu_2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    conv1_pool2d->setStrideNd(DimsHW{2, 2});

    // 第二层，2个卷积层和一个最大池化层
    IConvolutionLayer* conv2_conv2d_1 = network->addConvolutionNd(*conv1_pool2d->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2.0.weight"], emptywts);
    conv2_conv2d_1->setPaddingNd(DimsHW{1, 1});
    conv2_conv2d_1->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv2_bn2d_1 = addBatchNorm2d(network, weightMap, *conv2_conv2d_1->getOutput(0), "conv2.1", 1e-5);

    IActivationLayer* conv2_relu_1 = network->addActivation(*conv2_bn2d_1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2_conv2d_2 = network->addConvolutionNd(*conv2_relu_1->getOutput(0), 128, DimsHW{3, 3}, weightMap["conv2.3.weight"], emptywts);
    conv2_conv2d_2->setPaddingNd(DimsHW{1, 1});
    conv2_conv2d_2->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv2_bn2d_2 = addBatchNorm2d(network, weightMap, *conv2_conv2d_2->getOutput(0), "conv2.4", 1e-5);

    IActivationLayer* conv2_relu_2 = network->addActivation(*conv2_bn2d_2->getOutput(0), ActivationType::kRELU);

    IPoolingLayer* conv2_pool2d = network->addPoolingNd(*conv2_relu_2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    conv2_pool2d->setStrideNd(DimsHW{2, 2});

    // 第三层，3个卷积层和一个最大池化层
    IConvolutionLayer* conv3_conv2d_1 = network->addConvolutionNd(*conv2_pool2d->getOutput(0), 256, DimsHW{3, 3}, weightMap["conv3.0.weight"], emptywts);
    conv3_conv2d_1->setPaddingNd(DimsHW{1, 1});
    conv3_conv2d_1->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv3_bn2d_1 = addBatchNorm2d(network, weightMap, *conv3_conv2d_1->getOutput(0), "conv3.1", 1e-5);

    IActivationLayer* conv3_relu_1 = network->addActivation(*conv3_bn2d_1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv3_conv2d_2 = network->addConvolutionNd(*conv3_relu_1->getOutput(0), 256, DimsHW{3, 3}, weightMap["conv3.3.weight"], emptywts);
    conv3_conv2d_2->setPaddingNd(DimsHW{1, 1});
    conv3_conv2d_2->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv3_bn2d_2 = addBatchNorm2d(network, weightMap, *conv3_conv2d_2->getOutput(0), "conv3.4", 1e-5);

    IActivationLayer* conv3_relu_2 = network->addActivation(*conv3_bn2d_2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv3_conv2d_3 = network->addConvolutionNd(*conv3_relu_2->getOutput(0), 256, DimsHW{3, 3}, weightMap["conv3.6.weight"], emptywts);
    conv3_conv2d_3->setPaddingNd(DimsHW{1, 1});
    conv3_conv2d_3->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv3_bn2d_3 = addBatchNorm2d(network, weightMap, *conv3_conv2d_3->getOutput(0), "conv3.7", 1e-5);

    IActivationLayer* conv3_relu_3 = network->addActivation(*conv3_bn2d_3->getOutput(0), ActivationType::kRELU);

    IPoolingLayer* conv3_pool2d = network->addPoolingNd(*conv3_relu_3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    conv3_pool2d->setStrideNd(DimsHW{2, 2});

    // 第四层，3个卷积层和一个最大池化层
    IConvolutionLayer* conv4_conv2d_1 = network->addConvolutionNd(*conv3_pool2d->getOutput(0), 512, DimsHW{3, 3}, weightMap["conv4.0.weight"], emptywts);
    conv4_conv2d_1->setPaddingNd(DimsHW{1, 1});
    conv4_conv2d_1->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv4_bn2d_1 = addBatchNorm2d(network, weightMap, *conv4_conv2d_1->getOutput(0), "conv4.1", 1e-5);

    IActivationLayer* conv4_relu_1 = network->addActivation(*conv4_bn2d_1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv4_conv2d_2 = network->addConvolutionNd(*conv4_relu_1->getOutput(0), 512, DimsHW{3, 3}, weightMap["conv4.3.weight"], emptywts);
    conv4_conv2d_2->setPaddingNd(DimsHW{1, 1});
    conv4_conv2d_2->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv4_bn2d_2 = addBatchNorm2d(network, weightMap, *conv4_conv2d_2->getOutput(0), "conv4.4", 1e-5);

    IActivationLayer* conv4_relu_2 = network->addActivation(*conv4_bn2d_2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv4_conv2d_3 = network->addConvolutionNd(*conv4_relu_2->getOutput(0), 512, DimsHW{3, 3}, weightMap["conv4.6.weight"], emptywts);
    conv4_conv2d_3->setPaddingNd(DimsHW{1, 1});
    conv4_conv2d_3->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv4_bn2d_3 = addBatchNorm2d(network, weightMap, *conv4_conv2d_3->getOutput(0), "conv4.7", 1e-5);

    IActivationLayer* conv4_relu_3 = network->addActivation(*conv4_bn2d_3->getOutput(0), ActivationType::kRELU);

    IPoolingLayer* conv4_pool2d = network->addPoolingNd(*conv4_relu_3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    conv4_pool2d->setStrideNd(DimsHW{2, 2});

    // 第五层，3个卷积层和一个最大池化层
    IConvolutionLayer* conv5_conv2d_1 = network->addConvolutionNd(*conv4_pool2d->getOutput(0), 512, DimsHW{3, 3}, weightMap["conv5.0.weight"], emptywts);
    conv5_conv2d_1->setPaddingNd(DimsHW{1, 1});
    conv5_conv2d_1->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv5_bn2d_1 = addBatchNorm2d(network, weightMap, *conv5_conv2d_1->getOutput(0), "conv5.1", 1e-5);

    IActivationLayer* conv5_relu_1 = network->addActivation(*conv5_bn2d_1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv5_conv2d_2 = network->addConvolutionNd(*conv5_relu_1->getOutput(0), 512, DimsHW{3, 3}, weightMap["conv5.3.weight"], emptywts);
    conv5_conv2d_2->setPaddingNd(DimsHW{1, 1});
    conv5_conv2d_2->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv5_bn2d_2 = addBatchNorm2d(network, weightMap, *conv5_conv2d_2->getOutput(0), "conv5.4", 1e-5);

    IActivationLayer* conv5_relu_2 = network->addActivation(*conv5_bn2d_2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv5_conv2d_3 = network->addConvolutionNd(*conv5_relu_2->getOutput(0), 512, DimsHW{3, 3}, weightMap["conv5.6.weight"], emptywts);
    conv5_conv2d_3->setPaddingNd(DimsHW{1, 1});
    conv5_conv2d_3->setStrideNd(DimsHW{1, 1});

    IScaleLayer* conv5_bn2d_3 = addBatchNorm2d(network, weightMap, *conv5_conv2d_3->getOutput(0), "conv5.7", 1e-5);

    IActivationLayer* conv5_relu_3 = network->addActivation(*conv5_bn2d_3->getOutput(0), ActivationType::kRELU);

    IPoolingLayer* conv5_pool2d = network->addPoolingNd(*conv5_relu_3->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    conv5_pool2d->setStrideNd(DimsHW{2, 2});

    // reshape
    auto* reshape_layer = network->addShuffle(*conv5_pool2d->getOutput(0));
    reshape_layer->setReshapeDimensions(Dims32 {2, {-1, 512 * 7 * 7}});

    // 全连接层
    auto* fc_1_w  = network->addConstant(Dims32 {2, {4096, 512 * 7 * 7}}, weightMap["fc.0.weight"]);
    auto* fc_1_multiply = network->addMatrixMultiply(*reshape_layer->getOutput(0), MatrixOperation::kNONE, *fc_1_w->getOutput(0), MatrixOperation::kTRANSPOSE);
    auto* fc_1_b  = network->addConstant(Dims32 {2, {1, 4096}}, weightMap["fc.0.bias"]);
    auto* fc_1 = network->addElementWise(*fc_1_multiply->getOutput(0), *fc_1_b->getOutput(0), ElementWiseOperation::kSUM);
    auto* fc_1_relu = network->addActivation(*fc_1->getOutput(0), ActivationType::kRELU);

    auto* fc_2_w  = network->addConstant(Dims32 {2, {4096, 4096}}, weightMap["fc.3.weight"]);
    auto* fc_2_multiply = network->addMatrixMultiply(*fc_1_relu->getOutput(0), MatrixOperation::kNONE, *fc_2_w->getOutput(0), MatrixOperation::kTRANSPOSE);
    auto* fc_2_b  = network->addConstant(Dims32 {2, {1, 4096}}, weightMap["fc.3.bias"]);
    auto* fc_2 = network->addElementWise(*fc_2_multiply->getOutput(0), *fc_2_b->getOutput(0), ElementWiseOperation::kSUM);
    auto* fc_2_relu = network->addActivation(*fc_2->getOutput(0), ActivationType::kRELU);

    auto* fc_3_w  = network->addConstant(Dims32 {2, {classesNum, 4096}}, weightMap["fc.6.weight"]);
    auto* fc_3_multiply = network->addMatrixMultiply(*fc_2_relu->getOutput(0), MatrixOperation::kNONE, *fc_3_w->getOutput(0), MatrixOperation::kTRANSPOSE);
    auto* fc_3_b  = network->addConstant(Dims32 {2, {1, classesNum}}, weightMap["fc.6.bias"]);
    auto* fc_3 = network->addElementWise(*fc_3_multiply->getOutput(0), *fc_3_b->getOutput(0), ElementWiseOperation::kSUM);

    fc_3->getOutput(0)->setName(outputName);
    network->markOutput(*fc_3->getOutput(0));
}


int run()
{
    ICudaEngine *engine = nullptr;

    if (access(trtFile.c_str(), F_OK) == 0)
    {
        std::ifstream engineFile(trtFile, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return 1; }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return 1; }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 30);
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode)
        {
            config->setFlag(BuilderFlag::kFP16);
        }
        if (bINT8Mode)
        {
            config->setFlag(BuilderFlag::kINT8);
            int batchSize = 16;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, inputWidth, inputHeight, calibrationDataPath.c_str(), cacheFile.c_str());
            config->setInt8Calibrator(pCalibrator);
        }

        buildNetwork(network, profile, config);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        IRuntime* runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) { std::cout << "Failed building engine!" << std::endl; return 1; }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr)
        {
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setBindingDimensions(0, Dims32 {4, {1, 3, inputHeight, inputWidth}});

    std::vector<int> vTensorSize(2, 0);  // bytes of input and output
    for (int i = 0; i < 2; i++)
    {
        Dims32 dim = context->getBindingDimensions(i);
        int size = 1;
        for (int j = 0; j < dim.nbDims; j++)
        {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getBindingDataType(i));
    }

    // prepare input data and output data ---------------------------
    float inputData[3 * inputHeight * inputWidth];
    float outputData[classesNum];

    std::vector<std::string> file_names;
    if (read_files_in_dir(testDataPath.c_str(), file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    // inference
    int total_cost = 0;
    int img_count = 0;
    for (int i = 0; i < file_names.size(); i++)
    {
        std::string testImagePath = testDataPath + "/" + file_names[i];
        cv::Mat img = cv::imread(testImagePath, cv::IMREAD_COLOR);

        auto start = std::chrono::system_clock::now();
        imagePreProcess(img, inputData);  // put image data on inputData
        inference_one(context, inputData, outputData, vTensorSize);
        imagePostProcess(file_names[i], outputData);
        auto end = std::chrono::system_clock::now();

        total_cost += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        img_count++;
    }

    int avg_cost = total_cost / img_count;
    std::cout << "Total image num is: " << img_count;
    std::cout << " inference total cost is: " << total_cost << "ms";
    std::cout << " average cost is: " << avg_cost << "ms" << std::endl;

    return 0;
}

int main()
{
    CHECK(cudaSetDevice(0));
    run();
    return 0;
}
