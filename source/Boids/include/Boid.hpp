//
//  Boid.hpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 13.11.2020.
//

#ifndef Boid_hpp
#define Boid_hpp

#include <cstdio>
#include <opencv2/opencv.hpp>
#include "Vector.hpp"
#include <cstdlib>
#include "../../Helpers/include/MathHelper.hpp"

class Boid {
private:
    unsigned long id;
    cv::Point2f position;
    Vector velocity;

    bool operator == (const Boid &ref) const;

    unsigned long getID() const;


    cv::Point2f getDistanceTo(Boid boid);
    bool updatePosition();
    bool updatePosition(std::vector<cv::Point2f> points);
    friend std::ostream& operator<<(std::ostream& os, const Boid& boid);
protected:
public:
    Boid();
    bool updateVelocity(std::vector<cv::Point2f> points);
    static float getDistanceBetween(Boid firstBoid, Boid secondBoid);
    static Boid initWithinConstraint(int x, int y);
    static Boid initWithinConstraint(int maxX, int maxY, int margin);
    static Boid initAtPosition(float x, float y, float dx, float dy);
    std::string const&  to_str() const;
    cv::Point2f getPosition();
    const Vector& getVelocity() const {return velocity;};

    static Boid initWithinConstraint(int x, int y, int margin, float dx, float dy);
};

#endif /* Boid_hpp */
