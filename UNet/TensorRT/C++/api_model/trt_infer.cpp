#include "public.h"
#include "calibrator.h"
#include "mask2color.h"

using namespace nvinfer1;


const char *      inputName = "data";
const char *      outputName = "mask";
const int         classesNum = 32;
const int         inputHeight = 448;
const int         inputWidth = 448;
const int         outputSize = inputHeight * inputWidth;
const std::string wtsFile = "./para.wts";
const std::string trtFile = "./model.plan";
const std::string dataPath = "../../../../Camvid_segment_dataset";
const std::string valDataPath = dataPath + "/images/val";
const std::string testDataPath = dataPath + "/images/test";  // 用于推理
static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool        bFP16Mode = false;
// for INT8 mode
const bool        bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = valDataPath;  // 用于 int8 量化


// preprocess same as pytorch training
void imagePreProcess(cv::Mat& img, float* inputData)
{
    // resize
    cv::Mat resizeImg;
    cv::resize(img, resizeImg, cv::Size(inputWidth, inputHeight));

    // transpose((2, 0, 1)) and bgr to rgb
    uchar* uc_pixel = resizeImg.data;
    for (int i = 0; i < inputHeight * inputWidth; i++)
    {
        inputData[i] = ((float)uc_pixel[2]) / 255.0;
        inputData[i + inputHeight * inputWidth] = ((float)uc_pixel[1]) / 255.0;
        inputData[i + 2 * inputHeight * inputWidth] = ((float)uc_pixel[0]) / 255.0;
        uc_pixel += 3;
    }
}

void imagePostProcess(std::string file_name, int* outputs, int originHeight, int originWidth)
{
    cv::Mat mask(inputHeight, inputWidth, CV_8UC1);
    uchar* uc_pixel = mask.data;
    for (int i = 0; i < inputHeight * inputWidth; i++)
    {
        uc_pixel[i] = (uchar)outputs[i];
    }
    // resize
    cv::Mat resizeMask;
    cv::resize(mask, resizeMask, cv::Size(originWidth, originHeight), 0, 0, cv::INTER_NEAREST);
    // blur
    cv::Mat blurMask;
    cv::medianBlur(resizeMask, blurMask, 7);
    // mask to color
    cv::Mat colorImg = cv::Mat::zeros(originHeight, originWidth, CV_8UC3);
    toColor(blurMask, colorImg);
    // save segmentation result
    cv::imwrite("mask_" + file_name, colorImg);
}


void inference_one(IExecutionContext* context, float* inputData, int* outputData, std::vector<int>& vTensorSize)
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


IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
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


IActivationLayer* doubleConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".double_conv.0.weight"], emptywts);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".double_conv.1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".double_conv.3.weight"], emptywts);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".double_conv.4", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    return relu2;
}


IActivationLayer* down(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname)
{
    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});

    return doubleConv(network, weightMap, *pool1->getOutput(0), outch, lname + ".maxpool_conv.1");
}


IActivationLayer* up(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int inch, int outch, std::string lname)
{
    IDeconvolutionLayer* deconv = network->addDeconvolutionNd(input1, inch / 2, DimsHW{ 2, 2 }, weightMap[lname + ".up.weight"], weightMap[lname + ".up.bias"]);
    deconv->setStrideNd(DimsHW{ 2, 2 });

    int diffy = input2.getDimensions().d[2] - deconv->getOutput(0)->getDimensions().d[2];
    int diffx = input2.getDimensions().d[3] - deconv->getOutput(0)->getDimensions().d[3];
    // IPaddingLayer* pad = network->addPaddingNd(*deconv->getOutput(0), DimsHW{ diffy / 2, diffx / 2 }, DimsHW{ diffy - (diffy / 2), diffx - (diffx / 2) });
    Dims32 outputDims{4, {1, inch / 2, input2.getDimensions().d[2], input2.getDimensions().d[3]}};
    ISliceLayer* pad = network->addSlice(*deconv->getOutput(0), Dims32{4, {0, 0, - diffy / 2, -diffx / 2}}, outputDims, Dims32{4, {1, 1, 1, 1}});
    pad->setMode(SliceMode::kFILL);

    ITensor* inputTensors[] = { &input2, pad->getOutput(0) };
    IConcatenationLayer* concat = network->addConcatenation(inputTensors, 2);

    return doubleConv(network, weightMap, *concat->getOutput(0), outch, lname + ".conv");
}


void buildNetwork(INetworkDefinition* network, IOptimizationProfile* profile, IBuilderConfig* config, std::map<std::string, Weights>& weightMap)
{
    ITensor* inputTensor = network->addInput(inputName, DataType::kFLOAT, Dims32 {4, {-1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 3, inputHeight, inputWidth}});
    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 3, inputHeight, inputWidth}});
    config->addOptimizationProfile(profile);

    IActivationLayer* inc = doubleConv(network, weightMap, *inputTensor, 64, "inc");
    IActivationLayer* down1 = down(network, weightMap, *inc->getOutput(0), 128, "down1");
    IActivationLayer* down2 = down(network, weightMap, *down1->getOutput(0), 256, "down2");
    IActivationLayer* down3 = down(network, weightMap, *down2->getOutput(0), 512, "down3");
    IActivationLayer* down4 = down(network, weightMap, *down3->getOutput(0), 1024, "down4");
    IActivationLayer* up1 = up(network, weightMap, *down4->getOutput(0), *down3->getOutput(0), 1024, 512, "up1");
    IActivationLayer* up2 = up(network, weightMap, *up1->getOutput(0), *down2->getOutput(0), 512, 256, "up2");
    IActivationLayer* up3 = up(network, weightMap, *up2->getOutput(0), *down1->getOutput(0), 256, 128, "up3");
    IActivationLayer* up4 = up(network, weightMap, *up3->getOutput(0), *inc->getOutput(0), 128, 64, "up4");
    // output conv
    IConvolutionLayer* outConv = network->addConvolutionNd(*up4->getOutput(0), classesNum, DimsHW{1, 1}, weightMap["outc.conv.weight"], weightMap["outc.conv.bias"]);
    // add topk layer
    ITopKLayer* top1 = network->addTopK(*outConv->getOutput(0), TopKOperation::kMAX, 1, 1U << 1);
    top1->getOutput(1)->setName(outputName);
    network->markOutput(*top1->getOutput(1));
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
    int outputData[outputSize];  // using int. output is index

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
        int originHeight = img.rows;
        int originWidth = img.cols;

        auto start = std::chrono::system_clock::now();
        imagePreProcess(img, inputData);  // put image data on inputData
        inference_one(context, inputData, outputData, vTensorSize);
        auto end = std::chrono::system_clock::now();

        imagePostProcess(file_names[i], outputData, originHeight, originWidth);  // for visualization, not necessary

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
