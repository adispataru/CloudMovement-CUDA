//
//  FileHelper.cpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 12.11.2020.
//

#include "FileHelper.hpp"
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

using namespace std;

/**
 * Reads a file from its location.
 *
 * It diferentiates between flow (.npy) files and images (.jpg, .png).
 *
 * @param fileName a path to the file including filename and extension.
 * @return an object containing the read file.
 */
Mat FileHelper::readFile(string fileName) {
    Mat data;
//    cout << "Reading from: " << fileName << "\n";
    if(fileName.substr(fileName.find_last_of(".") + 1) == "npy") {
        data = cv::readOpticalFlow(fileName);
    } else { /// assume image format
        data = imread(fileName);
    }
    return data;
}

/**
 * Writes a file to a location.
 *
 * It diferentiates between flow (.npy) files and images (.jpg, .png.)
 *
 * @param fileName a path to the location where data is to be saved, including the filename and extension.
 * @param data the data that is to be saved.
 * @return true after completion (beta)...
 */
bool FileHelper::writeFile(string fileName, Mat data) {
    fs::create_directories(fileName.substr(0, fileName.find_last_of("\\/")));
    if(fileName.substr(fileName.find_last_of(".") + 1) == "npy") {
        cv::writeOpticalFlow(fileName, data);
    } else { /// assume image format
        imwrite(fileName, data);
    }
    return true;
}

/**
 * Reads multiple files from a location.
 *
 * The parameter should be of the form: "path/to/folder/ *.extension".
 *
 * @param path the path to the files that are to be read.
 * @return an array of objects containing the read files.
 */
vector<Mat> FileHelper::readFiles(string path) {
    vector<cv::String> fileNames;
    glob(path, fileNames, false);
    vector<Mat> data;
    if(path.substr(path.find_last_of(".") + 1) == "npy") {
        for (size_t k=0; k<fileNames.size(); ++k) {
            data.push_back(cv::readOpticalFlow(fileNames[k]));
        }
    } else { /// assume image format
        for (size_t k=0; k<fileNames.size(); ++k) {
            data.push_back(imread(fileNames[k]));
        }
    }
    return data;
}
