//
//  Scene.hpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 13.11.2020.
//

#ifndef Scene_hpp
#define Scene_hpp

#include <cstdio>
#include "Boid.hpp"
#include "../../Helpers/include/MathHelper.hpp"
#include "../../Helpers/include/FileHelper.hpp"
#include <experimental/filesystem>

class Scene {
private:
    std::vector<Boid> boids;
    std::string outputFolder;
    int sizeX, sizeY;
    static const float FIXED_RANGE;
    static const float COLLISION_RANGE;
    cv::Mat windMap;
    cv::Mat scene;
    int framesShown;
    bool saveSimulation;
    int framesSaved = 0;
    bool previewSimulation = false;

    bool addBoid(Boid boid);
    cv::Point2f getCenterOfMass();
    cv::Point2f getCenterOfMass(std::vector<Boid> boids);

    cv::Point2f rule1(Boid boid);
    cv::Point2f rule2(Boid boid);
    cv::Point2f rule3(Boid boid);
    cv::Point2f ruleOfWind(Boid boid);
    void clearScene();
    void drawScene();
    bool updateSimulation();
    Mat updateWindPosition(Mat windMap);

    Mat averageWindMap(Mat windMap, int radius = 5);
public:
    Scene(int sizeX, int sizeY);
    bool computeDifferenceOfWindMaps(Mat& first, Mat& second, Mat& result);
    std::vector<Vector> getWindVectors(cv::Point2f location);
    int getSizeX();
    int getSizeY();
    cv::Mat getWindMap();
    bool updateWindMap(cv::Mat newWindMap);
    bool updateWindMapUsingBoids(int neighborhoodRadius, std::string outputFolder);
    bool addRandomBoid();
    bool addBoid(int x, int y, int margin);
    int getBoidsCount();
    std::vector<Boid> getNeighbors(Boid boid, float range);
    std::vector<Boid> getAllBoids();
    bool runSimulation(int steps, bool preview = false);
    bool startSimulation(std::string outputFolder, int startIndex);

    bool updateUsingWeightedAverage();
};

#endif /* Scene_hpp */
