#include "public.h"
#include "calibrator.h"
#include "mask2color.h"
#include <NvOnnxParser.h>

using namespace nvinfer1;


const int         classesNum = 32;
const int         inputHeight = 448;
const int         inputWidth = 448;
const int         outputSize = inputHeight * inputWidth;
const std::string onnxFile = "./model.onnx";
const std::string trtFile = "./model.plan";
const std::string dataPath = "../../../../Camvid_segment_dataset";
const std::string valDataPath = dataPath + "/images/val";  // 用于 int8 量化
const std::string testDataPath = dataPath + "/images/test";  // 用于推理
// const std::string testImagePath = dataPath + "/test/roses_01.jpg";  // 用于推理
static Logger     gLogger(ILogger::Severity::kERROR);

// for FP16 mode
const bool        bFP16Mode = false;
// for INT8 mode
const bool        bINT8Mode = false;
const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = valDataPath;  // 用于 int8 量化


int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names)
{
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// preprocess same as pytorch training
void imagePreProcess(cv::Mat& img, float* inputData)
{
    // cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

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


ICudaEngine* getEngine()
{
    ICudaEngine* engine = nullptr;

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

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser->parseFromFile(onnxFile.c_str(), int(gLogger.reportableSeverity)))
        {
            std::cout << std::string("Failed parsing .onnx file!") << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i)
            {
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return nullptr;
        }
        std::cout << std::string("Succeeded parsing .onnx file!") << std::endl;

        ITensor* inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, inputHeight, inputWidth}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {4, 3, inputHeight, inputWidth}});
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {8, 3, inputHeight, inputWidth}});
        config->addOptimizationProfile(profile);

        // unmark output(original output is conv layer)
        ITensor* outputTensor = network->getOutput(0);
        network->unmarkOutput(*outputTensor);
        // add topk layer
        ITopKLayer* top1 = network->addTopK(*outputTensor, TopKOperation::kMAX, 1, 1U << 1);
        top1->getOutput(0)->setName("mask");
        network->markOutput(*top1->getOutput(1));

        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config);
        std::cout << "Succeeded building serialized engine!" << std::endl;

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
