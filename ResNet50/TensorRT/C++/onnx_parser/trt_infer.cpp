#include "public.h"
#include "calibrator.h"
#include <NvOnnxParser.h>

using namespace nvinfer1;


const int         inputHeight = 224;
const int         inputWidth = 224;
const std::string onnxFile = "./model.onnx";
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
