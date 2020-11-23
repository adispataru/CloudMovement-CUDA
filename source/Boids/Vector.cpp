//
//  Vector.cpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 13.11.2020.
//

#include "Vector.hpp"

Vector::Vector() {
    
}

cv::Point2f Vector::averageVectorDisplacement(std::vector<Vector> vectors) {
    std::vector<Point2f> points;
    for (Vector vector : vectors) {
        points.push_back(vector.getDisplacement());
    }
    cv::Point2f average = averagePoints(points);
    return average;
}
Vector Vector::initWithDisplacementAndPosition(cv::Point displacement, cv::Point position) {
    Vector vector;
    vector.origin = removePoints(position, displacement);
    vector.displacement = displacement;
    return vector;
}

float Vector::getEuclidianDistance(cv::Point2f origin, cv::Point2f destination) {
    return sqrt(pow(destination.x - origin.x, 2) + pow(destination.y - origin.y, 2));
}

cv::Point2f Vector::getOrigin() {
    return origin;
}

Vector::Vector(cv::Point2f origin, cv::Point2f position) {
    this->origin = origin;
    float displacementX = position.x - origin.x;
    float displacementY = position.y - origin.y;
    this->displacement = cv::Point2f(displacementX, displacementY);
}
cv::Point2f Vector::getDisplacement() {
    return displacement;
}
