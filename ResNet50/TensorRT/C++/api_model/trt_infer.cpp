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
    // std::cout << "len " << len << std::endl;

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


IActivationLayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + ".conv1.weight"], emptywts);
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + ".conv3.weight"], emptywts);
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4)
    {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + ".downsample.0.weight"], emptywts);
        conv4->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    else
    {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }

    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);

    return relu3;
}


void buildNetwork(INetworkDefinition* network, IOptimizationProfile* profile, IBuilderConfig* config, std::map<std::string, Weights>& weightMap)
{
    ITensor* inputTensor = network->addInput(inputName, DataType::kFLOAT, Dims32 {4, {-1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 3, inputHeight, inputWidth}});
    config->addOptimizationProfile(profile);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*inputTensor, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2, "layer2.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.2");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, "layer2.3");

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2, "layer3.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.2");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.3");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.4");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, "layer3.5");

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.2");

    IPoolingLayer* pool2 = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    // pool2->setStrideNd(DimsHW{1, 1});

    // reshape
    // IShuffleLayer* reshape_layer = network->addShuffle(*pool2->getOutput(0));
    // reshape_layer->setReshapeDimensions(Dims32 {2, {-1, 2048}});

    // IConstantLayer* fc_w  = network->addConstant(Dims32 {2, {classesNum, 2048}}, weightMap["fc.weight"]);
    // IMatrixMultiplyLayer* fc_multiply = network->addMatrixMultiply(*reshape_layer->getOutput(0), MatrixOperation::kNONE, *fc_w->getOutput(0), MatrixOperation::kTRANSPOSE);
    // IConstantLayer* fc_b  = network->addConstant(Dims32 {2, {1, classesNum}}, weightMap["fc.bias"]);
    // IElementWiseLayer* fc1 = network->addElementWise(*fc_multiply->getOutput(0), *fc_b->getOutput(0), ElementWiseOperation::kSUM);

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), classesNum, weightMap["fc.weight"], weightMap["fc.bias"]);

    fc1->getOutput(0)->setName(outputName);
    network->markOutput(*fc1->getOutput(0));
}


ICudaEngine* getEngine()
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
        if (engineString.size() == 0) { std::cout << "Failed getting serialized engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        IRuntime *runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
        if (engine == nullptr) { std::cout << "Failed loading engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded loading engine!" << std::endl;
    }
    else
    {
        IBuilder *            builder     = createInferBuilder(gLogger);
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        IOptimizationProfile* profile     = builder->createOptimizationProfile();
        IBuilderConfig *      config      = builder->createBuilderConfig();
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
        // load .wts
        std::map<std::string, Weights> weightMap = loadWeights(wtsFile);

        buildNetwork(network, profile, config, weightMap);

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*) (mem.second.values));
        }

        IRuntime* runtime {createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size());
        if (engine == nullptr) { std::cout << "Failed building engine!" << std::endl; return nullptr; }
        std::cout << "Succeeded building engine!" << std::endl;

        if (bINT8Mode && pCalibrator != nullptr)
        {
            delete pCalibrator;
        }

        std::ofstream engineFile(trtFile, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size());
        std::cout << "Succeeded saving .plan file!" << std::endl;
    }

    return engine;
}


int run()
{
    ICudaEngine* engine = getEngine();

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
