//
//  Scene.hpp
//  CloudMovement-CPP
//
//  Created by Adrian Spataru on 13.03.2021.
//

#ifndef CUDAScene_hpp
#define CUDAScene_hpp

#include "Boid.hpp"
#include "../../Helpers/include/MathHelper.hpp"
#include "../../Helpers/include/FileHelper.hpp"
//#include <experimental/filesystem>
//namespace fs = std::experimental::filesystem;
//#include <experimental/filesystem>
//#include "omp.h"
//#include <device_launch_parameters.h>



class CUDAScene {
private:
    std::vector<Boid> boids;
    std::string outputFolder;
    int sizeX, sizeY;
    static float FIXED_RANGE;
    static float COLLISION_RANGE;
    cv::Mat windMap;
    cv::Mat scene;
    int framesShown;
    bool saveSimulation;
    int framesSaved = 0;
    bool previewSimulation = false;

    bool addBoid(Boid boid);
    cv::Point2f getCenterOfMass();
    cv::Point2f getCenterOfMass(std::vector<Boid> boids);

//    cv::Point2f rule1(Boid boid);
//    cv::Point2f rule2(Boid boid);
//    cv::Point2f rule3(Boid boid);
//    cv::Point2f ruleOfWind(Boid boid);
    void clearScene();
    void drawScene();
    bool updateSimulation();
    bool updateSimulation2D();
    void updateWindPosition();
    void averageWindMap(int radius = 5);
public:
    CUDAScene(int sizeX, int sizeY);
    static bool computeDifferenceOfWindMaps(const Mat& first, const Mat& second, Mat& result);
//    std::vector<Vector> getWindVectors(cv::Point2f location);
    int getSizeX();
    int getSizeY();
    cv::Mat getWindMap();
    bool updateWindMap(const cv::Mat& newWindMap);
    bool updateWindMapUsingBoids(int neighborhoodRadius);
    bool addRandomBoid();
    bool addBoid(int x, int y, int margin, float dx, float dy);
    int getBoidsCount();
//    std::vector<Boid> getNeighbors(Boid boid, float range);
    std::vector<Boid> getAllBoids();
    bool runSimulation(int steps, bool preview = false);
    bool startSimulation(std::string outFolder, int startIndex, int neighRadius);

    void drawCloudMask(const Mat &mask, float threshold);

    void drawCloudMask(const Mat &mask, float threshold, int maxBoids);
    void drawCloudMaskDifference(const Mat &pred, const Mat &real, float threshold, int realThresh);
};

#endif /* CUDAScene_hpp */
