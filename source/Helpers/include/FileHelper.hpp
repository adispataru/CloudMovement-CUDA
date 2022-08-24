//
//  FileHelper.hpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 12.11.2020.
//

#ifndef FileHelper_hpp
#define FileHelper_hpp

#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

using namespace cv;

class FileHelper {

public:
    static Mat readFile(std::string fileName);
    static std::vector<Mat> readFiles(std::string folderPath);
    static bool writeFile(std::string fileName, Mat data);
};

#endif /* FileHelper_hpp */
