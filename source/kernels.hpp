#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "Helpers/include/FileHelper.hpp"
#include "Helpers/include/MathHelper.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>
//#include <cmath>
//#include "Boids/include/Scene.hpp"
#include "Boids/include/CUDAScene.hpp"
#include "Helpers/include/OpticalFlowService.hpp"

#ifndef _KERNEL_H
#define _KERNEL_H

using namespace cv;




void testCuda();

void simpletestCuda();

void programOptions(const char *string);

void argParse(int argc, const char **pString);


void computeMapeAndMAEOnGPU(Mat &forecastedWindMap, Mat &realWindMap, double *mape_error_p, double *mae_error_p);

void countTrueAndFalsePositivesAndNegativesOnGPU(Mat mat, Mat mat1, size_t *pDouble, size_t *pDouble1, size_t *pDouble2,
                                                 size_t *pDouble3, float threshold);


int runExperiments(int CONFIG = 0);

void createCloudMaskFromFlow();
void createCloudMasks();
void computeMAPE();

#endif