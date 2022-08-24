//
//  Vector.cpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 13.11.2020.
//

#include "Vector.hpp"
#include "MathHelper.hpp"

Vector::Vector() {
}

/**
 * Averages the dispalcement of multiple Vector objects.
 *
 * @param vectors an array of Vector objects for which the average is to be computed.
 * @return a point representing the average displacement of the Vector objects.
 */
cv::Point2f Vector::averageVectorDisplacement(std::vector<Vector> vectors) {
    std::vector<Point2f> points;
    for (Vector vector : vectors) {
        points.push_back(vector.getDisplacement());
    }
    cv::Point2f average = averagePoints(points);
    return average;
}

/**
 * Initialyzes a Vector object to the specified parameters.
 *
 * It computes the origin from the two values given as parameters.
 *
 * @param displacement the displacement the Vector object has.
 * @param position the current position of the Vector object.
 * @return a new Vector object.
 */
Vector Vector::initWithDisplacementAndPosition(cv::Point displacement, cv::Point position) {
    Vector vector;
    vector.origin = removePoints(position, displacement);
    vector.displacement = displacement;
    return vector;
}

/**
 * Computes the Euclidian distance between two points.
 *
 * @param origin the point from which the distance is computed.
 * @param destination the point to which the distance is computed.
 * @return a number representing the distance between the two points.
 */
float Vector::getEuclidianDistance(cv::Point2f origin, cv::Point2f destination) {
    return sqrt(pow(destination.x - origin.x, 2) + pow(destination.y - origin.y, 2));
}

cv::Point2f Vector::getOrigin() {
    return origin;
}

/**
 * Initialyzes a Vector object to the specified parameters.
 *
 * It computes the displacement from the two values given as parameters.
 *
 * @param origin the origin of the Vector object.
 * @param position the current position of the Vector object.
 * @return a new Vector object.
 */
Vector::Vector(cv::Point2f origin, cv::Point2f position) {
    this->origin = origin;
    float displacementX = position.x - origin.x;
    float displacementY = position.y - origin.y;
    this->displacement = cv::Point2f(displacementX, displacementY);
}

cv::Point2f Vector::getDisplacement() const {
    return displacement;
}
