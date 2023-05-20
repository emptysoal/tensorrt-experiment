#include <iostream>
#include <fstream>
#include <dirent.h>
#include <iterator>
#include <opencv2/dnn/dnn.hpp>

#include "calibrator.h"

using namespace nvinfer1;


int read_files_in_dir2(const char *p_dir_name, std::vector<std::string> &file_names)
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
cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h)
{
    // resize
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(input_w, input_h));
    // Type conversion, convert unsigned int 8 to float 32
    cv::Mat img_float;
    resize_img.convertTo(img_float, CV_32FC3, 1.f / 255.0);

    cv::Mat pr_img;
    pr_img = img_float.clone();

    return pr_img;
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batch_size, int input_w, int input_h, const char* img_dir, const char* calib_table_name, bool read_cache)
    : batch_size_(batch_size)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , read_cache_(read_cache)
{
    input_count_ = 3 * input_w * input_h * batch_size;
    cudaMalloc(&device_input_, input_count_ * sizeof(float));
    read_files_in_dir2(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    cudaFree(device_input_);
}

int Int8EntropyCalibrator2::getBatchSize() const noexcept
{
    return batch_size_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (img_idx_ + batch_size_ > (int)img_files_.size()) { return false; }

    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batch_size_; i++)
    {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + "/" + img_files_[i], cv::IMREAD_COLOR);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat pr_img = preprocess_img(temp, input_w_, input_h_);
        input_imgs_.push_back(pr_img);
    }
    img_idx_ += batch_size_;
    cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1.0 / 0.225, cv::Size(input_w_, input_h_), cv::Scalar(0.485, 0.456, 0.406), true, false);

    cudaMemcpy(device_input_, blob.ptr<float>(0), input_count_ * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) noexcept
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}
