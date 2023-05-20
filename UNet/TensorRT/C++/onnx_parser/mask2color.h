#ifndef MASK2COLOR_H
#define MASK2COLOR_H

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>


void setColor(std::map<uchar, std::vector<uchar>>& idx2colorMap)
{
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(0, std::vector<uchar>{0, 0, 0}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(1, std::vector<uchar>{0, 0, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(2, std::vector<uchar>{0, 0, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(3, std::vector<uchar>{0, 64, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(4, std::vector<uchar>{0, 128, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(5, std::vector<uchar>{0, 128, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(6, std::vector<uchar>{64, 0, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(7, std::vector<uchar>{64, 0, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(8, std::vector<uchar>{64, 0, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(9, std::vector<uchar>{64, 64, 0}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(10, std::vector<uchar>{64, 64, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(11, std::vector<uchar>{64, 128, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(12, std::vector<uchar>{64, 128, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(13, std::vector<uchar>{64, 192, 0}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(14, std::vector<uchar>{64, 192, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(15, std::vector<uchar>{128, 0, 0}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(16, std::vector<uchar>{128, 0, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(17, std::vector<uchar>{128, 64, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(18, std::vector<uchar>{128, 64, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(19, std::vector<uchar>{128, 128, 0}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(20, std::vector<uchar>{128, 128, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(21, std::vector<uchar>{128, 128, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(22, std::vector<uchar>{128, 128, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(23, std::vector<uchar>{192, 0, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(24, std::vector<uchar>{192, 0, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(25, std::vector<uchar>{192, 0, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(26, std::vector<uchar>{192, 64, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(27, std::vector<uchar>{192, 128, 64}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(28, std::vector<uchar>{192, 128, 128}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(29, std::vector<uchar>{192, 128, 192}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(30, std::vector<uchar>{192, 192, 0}) );
    idx2colorMap.insert( std::pair<uchar, std::vector<uchar>>(31, std::vector<uchar>{192, 192, 128}) );
}


void toColor(cv::Mat& mask, cv::Mat& colorImg)
{
    std::map<uchar, std::vector<uchar>> idx2colorMap;
    setColor(idx2colorMap);

    for (int row = 0; row < mask.rows; row++)
    {
        for (int col = 0; col < mask.cols; col++)
        {
            uchar& maskPixel = mask.at<uchar>(row, col);
            cv::Vec3b& colorPixel = colorImg.at<cv::Vec3b>(row, col);
            colorPixel[2] = idx2colorMap[maskPixel][0];  // R
            colorPixel[1] = idx2colorMap[maskPixel][1];  // G
            colorPixel[0] = idx2colorMap[maskPixel][2];  // B
        }
    }
}


#endif  // MASK2COLOR_H