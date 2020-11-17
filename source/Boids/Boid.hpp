//
//  Boid.hpp
//  CloudMovement-CPP
//
//  Created by Marius E. Penteliuc on 13.11.2020.
//

#ifndef Boid_hpp
#define Boid_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Vector.hpp"

class Boid {
private:
    unsigned long id;
    cv::Point2f position;
    Vector velocity;

    bool operator == (const Boid &ref) const;

    const long getID() const;

    static void initAtPosition(float x, float y);
    cv::Point2f getDistanceTo(Boid boid);
    friend std::ostream& operator<<(std::ostream& os, const Boid& boid);
protected:
public:
    Boid();
    static float getDistanceBetween(Boid firstBoid, Boid secondBoid);
    static Boid initWithinConstraint(int x, int y);
    std::string const&  to_str() const;
    cv::Point2f getPosition();
};

#endif /* Boid_hpp */
