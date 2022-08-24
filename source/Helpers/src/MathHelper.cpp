//
//  MathHelper.cpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 16.11.2020.
//

#include "MathHelper.hpp"
#include <ctime>
#include <cmath>

Point2f addPoints(Point2f v1, Point2f v2) {
    float x = v1.x + v2.x;
    float y = v1.y + v2.y;
    return Point2f(x, y);
}
static void remapNumbers(float &number) {
    if (number < 0.5 && number > -0.5) {
        number = 0;
    } else if (number >= 0.5) {
        number = 1;
    } else {
        number = -1;
    }
}

Point2f transformToSimpleDirection(Point2f point) {
    remapNumbers(point.x);
    remapNumbers(point.y);
    return point;
}
Point2f addPointsByPercentage(Point2f v1, float p1, Point2f v2, float p2) {
    float x = p1 * v1.x + p2 * v2.x;
    float y = p1 * v1.y + p2 * v2.y;
    return Point2f(x, y);
}
Point2f computeAPE(Point2f real, Point2f forecast) {
    return Point2f((abs(real.x - forecast.x) / real.x)*100, (abs(real.y - forecast.y) / real.y)*100);
//    return (abs(Vector::getEuclidianDistance(Point2f(0,0), real) - Vector::getEuclidianDistance(Point2f(0,0), forecast)) / Vector::getEuclidianDistance(Point2f(0,0), real)) * 100;
}
Point2f removePoints(Point2f v1, Point2f v2) {
    float x = v1.x - v2.x;
    float y = v1.y - v2.y;
    return Point2f(x, y);
}
Point2f removePoints(Point2f v1, float a) {
    float x = v1.x - a;
    float y = v1.y - a;
    return Point2f(x, y);
}
Point2f dividePoint(Point2f v1, float a) {
    float x = v1.x / a;
    float y = v1.y / a;
    return Point2f(x, y);
}

Point2f multiplyPoint(Point2f p, float f) {
    float x = p.x * f;
    float y = p.y * f;
    return Point2f(x, y);
}

float ruleOfThree(float part, float whole) {
    return part / whole;
}

Point2f averagePoints(std::vector<cv::Point2f> points) {
    cv::Point2f average;
    for (Point2f point: points) {
        average = addPoints(average, point);
    }
    return dividePoint(average, points.size());
}

Point2f averagePointsUsingWeights(Point2f p20, Point2f p80) {
    cv::Point2f average;
    float x = p20.x * 0.2 + p80.x * 0.8;
    float y = p20.y * 0.2 + p80.y * 0.8;
    average.x = x;
    average.y = y;

    return average;
}
