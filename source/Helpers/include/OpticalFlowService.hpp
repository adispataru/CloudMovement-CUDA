//
//  OpticalFlowService.hpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 12.11.2020.
//

#ifndef OpticalFlowService_hpp
#define OpticalFlowService_hpp

#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>
//#include <filesystem>



class OpticalFlowService {
private:
    static void drawOpticalFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, double, const cv::Scalar& color);
    static cv::Mat getOpticalFlowFarneback(cv::Mat firstImage, cv::Mat secondImage);
public:
    static cv::Mat overlayFlowLines(cv::Mat flow, cv::Mat image);
    OpticalFlowService();
    static int computeFlowForImages(std::string inputPath, std::string outputPath, std::string fileType, bool saveOverlays, bool saveFlows, bool previewOverlays);
    static cv::Mat averageFlows(std::string inputPath, size_t index = 0, size_t numberOfFlows = 0);
};

#endif /* OpticalFlowService_hpp */
