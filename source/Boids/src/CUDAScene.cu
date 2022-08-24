//
//  CUDAScene.cpp
//  CloudMovement-CPP
//
//  Created by Adrian Spataru on 13.03.2021.
//

#include "CUDAScene.hpp"

float CUDAScene::FIXED_RANGE = 5;
float CUDAScene::COLLISION_RANGE = 10;

using namespace std;

const char * updateType = "MAX";

#define DEBUG false

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void averageFloat2WithRadiusKernel(float2* output,
                                              float2* input,
                                              int width, int height, int radius)
{
    // Calculate normalized texture coordinates
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if(tx > width || ty > height)
        return;

    int x = (int) tx, y = (int) ty;
//    int colStartIndex = y - (radius - 1)/2;
//    if (colStartIndex < 0) colStartIndex = 0;
//    int rowStartIndex = x - (radius - 1)/2;
//    if (rowStartIndex < 0) rowStartIndex = 0;
//    int colEndIndex = y + (radius - 1)/2;
//    if (colEndIndex > width-1) colEndIndex = width-1;
//    int rowEndIndex = x + (radius - 1)/2;
//    if (rowEndIndex > height-1) rowEndIndex = height-1;

    float2 currentPoint = input[y*width+x];//tex2D<float2>(texObj, x, y);
    float2 fastestVector = {0,0};
    float2 average = {0,0};
    int elems = 0;
    float currentDisplacement = sqrt(currentPoint.x * currentPoint.x + currentPoint.y *currentPoint.y);
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            if(x+j < 0)
                continue;
            if(x+j >= width)
                break;
            if(y+i < 0)
                continue;
            if(y+i >= height)
                break;

            float2 fxy = input[(y+i) * width + (x+j)];//tex2D<float2>(texObj, x+j, y+i);

            float neighborDisplacement = sqrt(fxy.x * fxy.x + fxy.y *fxy.y);
            if (neighborDisplacement >= currentDisplacement) {
                fastestVector = fxy;
            }
            // addPoints(average, fxy);
            average.x = average.x + fxy.x;
            average.y = average.y + fxy.y;
            elems++;
        }
    }
//            currentPoint = averagePointsUsingWeights(currentPoint, fastestVector);

//    currentPoint = dividePoint(average, data.rows * data.cols);
    if(elems!=0) {
//        currentPoint.x += average.x / 1.0f * elems;
//        currentPoint.y += average.y / 1.0f * elems;
        average.x = average.x / (1.0f*elems);
        average.y = average.y / (1.0f*elems);

        currentPoint.x += fastestVector.x/2.0f;
        currentPoint.y += fastestVector.y/2.0f;
    }
    currentPoint.x += average.x;
    currentPoint.y += average.y;
//            50% weight on the fastest wind vector
//currentPoint = dividePoint(addPoints(currentPoint, fastestVector), 2);



    // write to global memory
    output[y * width + x] = currentPoint;
//    output[y * width + x] = {0, 0};
}

__global__ void matrixDifferenceUsingTexturesKernel(float2* comparison, float2* output,
                                              cudaTextureObject_t texObj,
                                              int width, int height)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > width || y > height)
        return;

    float2 currentPoint = tex2D<float2>(texObj, x, y);
    float2 comp = comparison[y * width + x];
    currentPoint.x -= comp.x;
    currentPoint.y -= comp.y;


    // write to global memory
    output[y * width + x] = currentPoint;
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void updateWindPositionKernelUsingMAX(float2* output,
                                                 float2*input,
                                                 int width, int height)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > width || y > height)
        return;

    float2 fxy = input[y*width+x]; //tex2D<float2>(texObj, 130.0f, 130.0f);

    float2 newPoint = {(float)x+fxy.x, (float)y+fxy.y};

    if (newPoint.x >= 0 && newPoint.y >=0 && newPoint.x < (float)width && newPoint.y < (float)height) {
        int newX = (int) ceil(newPoint.x);
        int newY = (int) ceil(newPoint.y);

        //this code adds together winds landing on the same pixel
//        atomicAdd(&(output[newY*width + newX].x), fxy.x);
//        atomicAdd(&(output[newY*width + newX].y), fxy.y);


        //this code chooses the max wind if multiple are landing on the same pixel
        atomicMax(&(output[newY*width + newX].x), fxy.x);
        atomicMax(&(output[newY*width + newX].y), fxy.y);
//        output[newY*width + newX] = {landingWind.x + fxy.x, landingWind.y + fxy.y};

    }
//    else{
//        output[y*width + x] = {0,0};
//    }
    //            cout << "changing " << updatedwindmap.at<Point2f>(newPoint) << " into " << fxy << endl;
}

__global__ void updateWindPositionKernelUsingAverage(float2* output, float* displacement,
                                                 float2*input,
                                                 int width, int height)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > width || y > height)
        return;

    float2 fxy = input[y*width+x]; //tex2D<float2>(texObj, 130.0f, 130.0f);

    float2 newPoint = {(float)x+fxy.x, (float)y+fxy.y};

    if (newPoint.x >= 0 && newPoint.y >=0 && newPoint.x < (float)width && newPoint.y < (float)height) {
        int newX = (int) ceil(newPoint.x);
        int newY = (int) ceil(newPoint.y);

        //this code adds together winds landing on the same pixel
        atomicAdd(&(output[newY*width + newX].x), fxy.x);
        atomicAdd(&(output[newY*width + newX].y), fxy.y);
        //instead of counting the vectors, add the displacement of vector = sqrt(fxy.x^2 + fxy.y^2)
        float disp = sqrt(fxy.x * fxy.x + fxy.y*fxy.y);

        ///this will divide the point by the sum of the displacements
        /// therefore it's value will be between 0 and 1.
//        atomicAdd(&(displacement[newY * width + newX]), disp);

        ///this will make the normal average
        atomicAdd(&(displacement[newY * width + newX]), 1);

        //this code chooses the max wind if multiple are landing on the same pixel
//        atomicMax(&(output[newY*width + newX].x), fxy.x);
//        atomicMax(&(output[newY*width + newX].y), fxy.y);
//        output[newY*width + newX] = {landingWind.x + fxy.x, landingWind.y + fxy.y};

    }
//    else{
//    output[y*width + x] = {0,0};
//    }
    //            cout << "changing " << updatedwindmap.at<Point2f>(newPoint) << " into " << fxy << endl;
}

__global__ void applyBoidsVelocityRuleKernel(float2* input, float2* output,
                                             float2* windTexture,
                                             size_t size, int width, int height, int FIXED_RANGE)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size) {

//        points.push_back(ruleOfWind(boid));
//        std::vector<Vector> winds = getWindVectors(boid.getPosition());
//        Point2f location = boid.getPosition();
        float2 location = input[index];
        if(location.x < 0 || location.y < 0 || location.x >= 1920 || location.y >=1080 ){
            output[index] = location;
            return;
        }

//        std::vector<Vector> closeWindVectors;
        float2 average = {0,0};
        size_t count = 0;
        for (int i = -FIXED_RANGE; i <= FIXED_RANGE; ++i) {
            for (int j = -FIXED_RANGE; j <= FIXED_RANGE; ++j) {
                int yAxis = (int)(location.y + j);
                int xAxis = (int)(location.x + i);
                if (yAxis < 0)
                    continue;
                if(yAxis >= height)
                    break;
                if (xAxis < 0)
                    continue;
                if(xAxis >= width)
                    break;

                float2 displacement = windTexture[yAxis*width + xAxis];
//                this is unwrapped by creating the vector which holds the dispacement and location
//                however, this vector seems not needed, since the average only accounts for the computed displacement
//                Vector vector = Vector::initWithDisplacementAndPosition(displacement, location);
                //this is unwrapped by computing the substracted field based on the removePoints function from Vector
                // removePoints(position, substracted)
//                float2 substracted;
//                substracted.x = location.x - displacement.x;
//                substracted.y = location.y - displacement.y;
//                //first two are the origin and the next two are the displacement
//                float4 vector = {substracted.x, substracted.y, displacement.x, displacement.y};
                average.x += displacement.x;
                average.y += displacement.y;
                count++;

//                instead of adding all vectors to the closeWindVectors, we use the average float2 to accumulate
//                closeWindVectors.push_back(vector);
                // Rule 2 was commented out, but it is implemented bellow

            }
        }
        if(count != 0) {
//            average.x /= count;
//            average.y /= count; I SPENT 5 HOURS debugging why it worked with a generated flow map but
//                                it didn't work with the flow maps reads from *npy files.
//                                Still wondering why the code below fixes the issue.

            average.x = average.x / (1.0f*count);
            average.y = average.y / (1.0f*count);
        }
        location.x += average.x;
        location.y += average.y;
        output[index] = location;

    }

}

__global__ void applyBoidsVelocityAndColisionRulesKernel(float2* input, float2* output, size_t size,
                                                         float2* windTexture, float2* boidsVelocity,
                                                         int width, int height, int FIXED_RANGE, float COLLISION_RANGE)
{
    // Calculate normalized texture coordinates
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size) {

//        points.push_back(ruleOfWind(boid));
//        std::vector<Vector> winds = getWindVectors(boid.getPosition());
//        Point2f location = boid.getPosition();
        float2 location = input[index];
//        std::vector<Vector> closeWindVectors;
        float2 average = {0,0};
        size_t count = 0;
        size_t countBoids = 0;
        float2 collisionDistance = {0,0};
        for (int i = -FIXED_RANGE; i <= FIXED_RANGE; ++i) {
            for (int j = -FIXED_RANGE; j <= FIXED_RANGE; ++j) {
                int yAxis = (int) (location.y + j);
                int xAxis = (int) (location.x + i);
                if (yAxis < 0)
                    continue;
                if(yAxis >= height)
                    break;
                if (xAxis < 0)
                    continue;
                if(xAxis > width)
                    break;
                float2 displacement = windTexture[yAxis*width+xAxis];
//                this is unwrapped by creating the vector which holds the dispacement and location
//                however, this vector seems not needed, since the average only accounts for the computed displacement
//                Vector vector = Vector::initWithDisplacementAndPosition(displacement, location);
                //this is unwrapped by computing the removed field based on the removePoints function from Vector
                // removePoints(position, removed)
                //float4 vector = {removed.x, removed.y, displacement.x, displacement.y};
                average.x += displacement.x;
                average.y += displacement.y;
                count++;

//                instead of adding all vectors to the closeWindVectors, we use the average float2 to accumulate
//                closeWindVectors.push_back(vector);

                //This is rule2
                int bIndex = (int) yAxis * width + (int) xAxis;

                float2 boidVel = boidsVelocity[bIndex];


                if(abs(boidVel.x) > 0 && abs(boidVel.y) > 0){
                    //this is a valid boid

//                    average.x += boidVel.x;
//                    average.y += boidVel.y;
//                    count++;


                    float proximity = sqrt(pow((xAxis - location.x), 2.0f) + pow((yAxis - location.y), 2.0f));
                    if (abs(proximity) < COLLISION_RANGE) {
                        //removePoints(collisionDistance, proximity);
                        proximity = 1;
                        if(location.x < xAxis){
//                            collisionDistance.x -= proximity;
                            collisionDistance.x -= proximity;
                        }else{
                            collisionDistance.x += proximity;

                        }

                        if(location.y < yAxis) {
                            collisionDistance.y -= proximity;
                        }else{
                            collisionDistance.y += proximity;
                        }
                        countBoids++;
                    }
                }
            }
        }
        if(count) {
            average.x /= count;
            average.y /= count;
//            if(countBoids){
//                collisionDistance.x /= countBoids;
//                collisionDistance.y /= countBoids;
//                average.x += collisionDistance.x;
//                average.y += collisionDistance.y;
//                average.x /= 2;
//                average.y /= 2;
//            }
        }
        location.x += average.x;
        location.y += average.y;
        output[index] = location;
    }

}

__global__ void applyBoidsVelocityAndColisionRulesKernel2D(float2* input, float2* output, size_t size, size_t matLen,
                                                           float2* windTexture, float2* boidsVelocity,
                                                           int width, int height, int FIXED_RANGE, float COLLISION_RANGE)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index = y*matLen+x;

    if(index < size) {

//        points.push_back(ruleOfWind(boid));
//        std::vector<Vector> winds = getWindVectors(boid.getPosition());
//        Point2f location = boid.getPosition();
        float2 location = input[index];
//        std::vector<Vector> closeWindVectors;
        float2 average = {0,0};
        size_t count = 0;
        size_t countBoids = 0;
        float2 collisionDistance = {0,0};
        for (int i = -FIXED_RANGE; i <= FIXED_RANGE; ++i) {
            for (int j = -FIXED_RANGE; j <= FIXED_RANGE; ++j) {
                int yAxis = (int) (location.y + j);
                int xAxis = (int) (location.x + i);
                if (yAxis < 0)
                    continue;
                if(yAxis >= height)
                    break;
                if (xAxis < 0)
                    continue;
                if(xAxis > width)
                    break;

                //displacement due to wind
                float2 displacement = windTexture[yAxis * width + xAxis];
//                this is unwrapped by creating the vector which holds the dispacement and location
//                however, this vector seems not needed, since the average only accounts for the computed displacement
//                Vector vector = Vector::initWithDisplacementAndPosition(displacement, location);
                //this is unwrapped by computing the removed field based on the removePoints function from Vector
                // removePoints(position, removed)
                //float4 vector = {removed.x, removed.y, displacement.x, displacement.y};
                average.x += displacement.x;
                average.y += displacement.y;
                count++;

//                instead of adding all vectors to the closeWindVectors, we use the average float2 to accumulate
//                closeWindVectors.push_back(vector);

                //This is velocity matching and colission avoidance
                int bIndex = (int) yAxis * width + (int) xAxis;

                float2 boidVel = boidsVelocity[bIndex];


                if(abs(boidVel.x) > 0 && abs(boidVel.y) > 0){
                    //this is a valid boid

                    average.x += boidVel.x;
                    average.y += boidVel.y;
                    count++;


                    float proximity = sqrt(pow((xAxis - location.x), 2.0f) + pow((yAxis - location.y), 2.0f));
                    if (abs(proximity) < COLLISION_RANGE) {
                        //removePoints(collisionDistance, proximity);
                        proximity = 1;
                        if(location.x < xAxis){
//                            collisionDistance.x -= proximity;
                            collisionDistance.x -= proximity;
                        }else{
                            collisionDistance.x += proximity;

                        }

                        if(location.y < yAxis) {
                            collisionDistance.y -= proximity;
                        }else{
                            collisionDistance.y += proximity;
                        }
                        countBoids++;
                    }
                }
            }
        }
        if(count) {
            average.x /= count;
            average.y /= count;
//            if(countBoids){
//                collisionDistance.x /= countBoids;
//                collisionDistance.y /= countBoids;
//                average.x += collisionDistance.x;
//                average.y += collisionDistance.y;
//                average.x /= 2;
//                average.y /= 2;
//            }
        }
        location.x += average.x;
        location.y += average.y;
        output[index] = location;
    }


}

__global__ void applyScaledBoidsVelocityAndColisionRulesKernel2D(float2* input, float2* output, size_t size, size_t matLen,
                                                                 float2* windTexture, float2* boidsVelocity, float* boids_mag,
                                                                 int width, int height, int FIXED_RANGE, float COLLISION_RANGE)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;



    unsigned int index = y*matLen+x;

    if(index < size) {

//        points.push_back(ruleOfWind(boid));
//        std::vector<Vector> winds = getWindVectors(boid.getPosition());
//        Point2f location = boid.getPosition();
        float2 location = input[index];
//        std::vector<Vector> closeWindVectors;
        float2 average = {0, 0};
//        float2 averageNeg = {0, 0};
        float2 wind = {0, 0};
        size_t count = 0;
        size_t countBoids = 0;
        float2 collisionDistance = {0, 0};

        int boidX = (int) location.x;
        int boidY = (int) location.y;

        //displacement due to wind
        float2 displacement = windTexture[boidY * width + boidX];

        //the displacement due to the wind on this position
        wind.x += displacement.x;
        wind.y += displacement.y;

        if (boidX < 0 || boidX >= width || boidY < 0 || boidY >= height){
            output[index] = location;
            return;
        }



        for (int i = -FIXED_RANGE; i <= FIXED_RANGE; ++i) {
            for (int j = -FIXED_RANGE; j <= FIXED_RANGE; ++j) {


                int yAxis = boidY + j;
                int xAxis = boidX + i;
                if (yAxis < 0 || yAxis >= height || xAxis < 0 || xAxis >= width)
                    continue;




//                instead of adding all vectors to the closeWindVectors, we use the average float2 to accumulate
//                closeWindVectors.push_back(vector);
                int b2X = (int) xAxis;
                int b2Y = (int) yAxis;


                //displacement due to wind
//                if(i!=0 && j != 0) {

                displacement = windTexture[b2Y * width + b2X];



                //This is velocity matching and colission avoidance
                int bIndex = b2Y * width + b2X;

                float2 boidVel = boidsVelocity[bIndex];
                float mag = boids_mag[bIndex];
                if (mag > 0) {
                    //this is a valid boid or set of boids that have been averaged
                    //add its velocity to the average
                    //the velocity of the boid is scaled with respect to the distance of this boid
//                    average.x += boidVel.x / abs(j/2 +1);
//                    average.y += boidVel.y / abs(i/2 +1);

                    average.x += boidVel.x;
                    average.y += boidVel.y;
                    count += mag;

                    float proximity = (float) sqrt(pow((b2X - boidX), 2.0f) + pow((b2Y - boidY), 2.0f));
                    if (abs(proximity) < COLLISION_RANGE) {
                        //removePoints(collisionDistance, proximity);
//                        proximity = 1;
                        if (boidX < xAxis) {
//                            collisionDistance.x -= proximity;
                            collisionDistance.x -= proximity;
                        } else {
                            collisionDistance.x += proximity;

                        }

                        if (boidY < yAxis) {
                            collisionDistance.y -= proximity;
                        } else {
                            collisionDistance.y += proximity;
                        }
                        countBoids++;
                    }
                }
            }
        }
        if (count) {
            average.x /= count;
            average.y /= count;
//            if(countBoids){
//                collisionDistance.x /= countBoids;
//                collisionDistance.y /= countBoids;
//                average.x += collisionDistance.x;
//                average.y += collisionDistance.y;
//                average.x /= 2;
//                average.y /= 2;
//            }
        }

        // Only do this if wind is different than 0
        // actually, if wind is 0 it's ok for the equation
        float avgMag = sqrt(average.x * average.x + average.y * average.y);
        float windMag = sqrt(wind.x * wind.x + wind.y * wind.y);
        average.x = (avgMag * average.x + windMag * wind.x)  / (avgMag + windMag);
        average.y = (avgMag * average.y + windMag * wind.y) / (avgMag + windMag);

//        average.y = (average.y + wind.y)*0.5f;

        location.x += average.x;
        location.y += average.y;
//        location.x += wind.x;
//        location.y += wind.y;
        output[index] = location;
    }


}



/**
 * Constructor that initializes a new CUDAScene to the specified parameters.
 *
 * @param sizeX the width of the scene.
 * @param sizeY the height of the scene.
 */
CUDAScene::CUDAScene(int sizeX, int sizeY) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->scene = Mat::zeros( sizeY, sizeX, CV_8UC3 );
    this->framesShown = 0;
    this->saveSimulation = false;
    this->outputFolder = "./scene_output";
}

/**
 * Creates a new boid objects and puts it in the CUDAScene at a random location.
 *
 * @return true after completion (beta)...
 */
bool CUDAScene::addRandomBoid() {
    int initialSize = static_cast<int>(boids.size());
    boids.push_back(Boid::initWithinConstraint(sizeX, sizeY));
    if (initialSize < boids.size()) {
        return true;
    }
    return false;
}

bool CUDAScene::addBoid(int x, int y, int margin, float dx, float dy) {
    int initialSize = static_cast<int>(boids.size());
    if(margin == 0) {
        boids.push_back(Boid::initAtPosition(x, y, dx, dy));
    }else{
        boids.push_back(Boid::initWithinConstraint(x, y, margin, dx, dy));
    }
    if (initialSize < boids.size()) {
        return true;
    }
    return false;
}

int CUDAScene::getBoidsCount() {
    return static_cast<int>(boids.size());
}

int CUDAScene::getSizeX() {
    return sizeX;
}

int CUDAScene::getSizeY() {
    return sizeY;
}

/**
 * Computes the center of mass for all the boids in the scene.
 *
 * @return a point representing the center of mass for all boids.
 */
cv::Point2f CUDAScene::getCenterOfMass() {
    return getCenterOfMass(boids);
}

/**
 * Computes the center of mass for an array of boids.
 *
 * @param boids an array of boids for which the center of mass is to be computed.
 * @return a point representing the center of mass for given boids.
 */
cv::Point2f CUDAScene::getCenterOfMass(std::vector<Boid> boids) {
    cv::Point2f centerOfMass;
    for (Boid boid : boids) {
        centerOfMass = addPoints(centerOfMass, boid.getPosition());
    }
    return dividePoint(centerOfMass, boids.size());
}

/**
 * Returns all the boid objects that exist in the scene.
 *
 * @return an array with all boid objects in the scene.
 */

std::vector<Boid> CUDAScene::getAllBoids() {
    return boids;
}


/**
 * Updates the wind map used by the ruleOfWind() function.
 *
 * @param newWindMap an object that contains a flow of vectors.
 * @return true after completion (beta)...
 */
bool CUDAScene::updateWindMap(const cv::Mat& newWindMap) {
    newWindMap.copyTo(windMap);
    return true;
}




/// Updates the position of wind vectors by their velocity and direction of movement using CUDA.
/// Internal state stored in windMap is the collection of vectors to be updated.
void CUDAScene::updateWindPosition() {

    int width = windMap.cols;
    int height = windMap.rows;

    float2* cuArray;
    float2* output;
    gpuErrchk(cudaMalloc(&output, width * height * sizeof(float2)));
    gpuErrchk( cudaMalloc(&cuArray, width*height*sizeof(float2)));
    // Copy to device memory some data located at address h_data
    // in host memory
    size_t size = height*width;
    float2 *h_data = (float2*) malloc(height*width*sizeof (float2));
    float2* it = h_data;
    for(int i = 0; i < windMap.rows; i++){
        for(int j = 0; j < windMap.cols; j++){
            auto p = windMap.at<Point2f>(i, j);
            *it = {(float)p.x, (float)p.y};
            it++;
        }
    }

    gpuErrchk(cudaMemcpy(cuArray, h_data, size*sizeof(float2),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(output, 0, size*sizeof(float2)));

    gpuErrchk(cudaMemcpy(output, h_data, size*sizeof(float2), cudaMemcpyHostToDevice));

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);


    float *count;
    if(!strcmp(updateType, "AVERAGE")) {

        gpuErrchk(cudaMalloc(&count, width*height*sizeof(float)));
        gpuErrchk(cudaMemset(count, 0, width*height*sizeof(float)));
        //TODO Check if averaging the winds using displacement solves the precision issue
        //FIXED: this will just mess up the wind values to [0..1] and we cannot detect clouds anymore

        updateWindPositionKernelUsingAverage<<<dimGrid, dimBlock>>>(output, count, cuArray, width, height);

    }else{
        updateWindPositionKernelUsingMAX<<<dimGrid, dimBlock>>>(output, cuArray, width, height);
    }

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(cudaMemcpy(h_data, output, size*sizeof(float2),cudaMemcpyDeviceToHost));

    // Free device memory
    gpuErrchk(cudaFree(cuArray));
    gpuErrchk(cudaFree(output));

    it = h_data;
    for(int i = 0; i < windMap.rows; i++){
        for(int j = 0; j < windMap.cols; j++){
            auto p = windMap.at<Point2f>(i, j);
            windMap.at<Point2f>(i,j).x = it->x;
            windMap.at<Point2f>(i,j).y = it->y;
            it++;
        }
    }

    free(h_data);

    if(!strcmp(updateType, "AVERAGE")){
        float* count_h = (float*) malloc(width*height*sizeof(float));
        gpuErrchk(cudaMemcpy(count_h, count, width*height*sizeof(float), cudaMemcpyDeviceToHost));
        float *iit = count_h;
        for(int i = 0; i < windMap.rows; i++){
            for(int j = 0; j < windMap.cols; j++){
                auto p = windMap.at<Point2f>(i, j);
                if(*iit > 0) {
                    windMap.at<Point2f>(i, j).x /= (1.0 * *iit);
                    windMap.at<Point2f>(i, j).y /= (1.0 * *iit);
                }
                iit++;
            }
        }
        gpuErrchk(cudaFree(count));
        free(count_h);
    }

}


/// Averages the velocity and direction of neighboring groups of vectors using CUDA.
/// Makes use of the @... CUDA Kernel
/// The average is calculated between the vector and its neighbor with has the largest displacement value using weights of 20% and 80% respectively.
///
/// If radius extends beyond the boundary, missing neighbors will be ignored.
/// Internal state stored in windMap is the collection of vectors to be averaged.
/// @param radius the neighborhood window for the average, default is 5.
void CUDAScene::averageWindMap(int radius) {
//    Mat& input = this->windMap;
    int width = windMap.cols;
    int height = windMap.rows;

    float2* input;
    float2* output;
    gpuErrchk(cudaMalloc(&output, width * height * sizeof(float2)));
    gpuErrchk( cudaMalloc(&input, width * height * sizeof(float2)));
    gpuErrchk(cudaMemset(output, 0, width*height*sizeof(float2)));
    // Copy to device memory some data located at address h_data
    // in host memory
    size_t size = height*width;
    float2 *h_data = (float2*) malloc(height*width*sizeof (float2));
    float2* it = h_data;
    for(int i = 0; i < windMap.rows; i++){
        for(int j = 0; j < windMap.cols; j++){
            auto p = windMap.at<Point2f>(i, j);
            *it = {(float)p.x, (float)p.y};
            it++;
        }
    }

    gpuErrchk(cudaMemcpy(input, h_data, size * sizeof(float2), cudaMemcpyHostToDevice));



    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    averageFloat2WithRadiusKernel<<<dimGrid, dimBlock>>>(output,
                                                         input, width, height, radius);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_data, output, size*sizeof(float2),cudaMemcpyDeviceToHost));

    // Free device memory
    gpuErrchk(cudaFree(input));
    gpuErrchk(cudaFree(output));


//    Mat out = input;
//    Mat& out = input;
    it = h_data;
    for(int i = 0; i < windMap.rows; i++){
        for(int j = 0; j < windMap.cols; j++){
//            j == 130 && cout << i << " result: " << it->x << ", " << it->y << "\n";
            windMap.at<Point2f>(i,j).x = it->x;
            windMap.at<Point2f>(i,j).y = it->y;
            it++;
        }
    }

    free(h_data);
//    return out;
}

/**
 * Updates the wind map used by the ruleOfWind() function according to Boids Algorithm.
 *
 * @return true after completion (beta)...
 */
bool CUDAScene::updateWindMapUsingBoids(int neighbordhoodRadius) {
    cout << "Updating wind position\n";
    updateWindPosition();
//    if(strcmp(updateType, "AVERAGE") != 0) {
//        cout << "Averaging wind map with radius:" << neighbordhoodRadius << "\n";
//        averageWindMap(neighbordhoodRadius);
//    }
    cout << "Finished!\n";
    return true;
}

/**
 * Returns the wind map that is used by the ruleOfWind() function.
 * @return an object that contains a flow of vectors.
 */
cv::Mat CUDAScene::getWindMap() {
    return windMap;
}


/**
 * Updates the simulation by applying rules to each boid's motion using CUDA.
 * These rules are combined to create target points towards each boid object will navigate.
 *
 * @return true after completion (beta)...
 */
bool CUDAScene::updateSimulation() {

    //boids are given as a 1D float2 array and the Vectors are given as a 2D float4 texture
    //each thread processes one boid and computes its new velocity and stores it in
    //a 1D float2 array with each element on the original poisition.

//    Mat& windMap = this->windMap;
    int width = windMap.cols;
    int height = windMap.rows;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 32, 0, 0,
                                  cudaChannelFormatKindFloat);

    // Copy to device memory some data located at address h_wind
    // in host memory
    size_t size = height*width;
    float2* cuArray;
    gpuErrchk(cudaMalloc(&cuArray, size*sizeof(float2)));

    float2 *h_wind = (float2*) malloc(size * sizeof(float2));
    float2* it = h_wind;
    for(int i = 0; i < windMap.rows; i++){
        for(int j = 0; j < windMap.cols; j++){
            auto p = windMap.at<Point2f>(i, j);
//            cout << "**have point : " << p.x << ", "<<  p.y << "\n";
            *it = {(float)p.x, (float)p.y};
//            cout << "--wrote point in kernel: " << it->x<< ", " << it->y << "\n";
            it++;
        }
    }


    gpuErrchk(cudaMemcpy(cuArray, h_wind, size*sizeof(float2), cudaMemcpyHostToDevice));

    // Specify texture


    // Allocate the boids arrays
    size_t numBoids = boids.size();
    float2* localBoids = (float2*) malloc(numBoids * sizeof(float2));
    float2* itB = localBoids;
    for (Boid& boid : boids){
        const Point2f& p = boid.getPosition();
//        cout << "**have point : " << p.x << ", "<<  p.y << "\n";
        *itB = {(float)p.x, (float)p.y};
//        cout << "--wrote point in kernel: " << itB->x<< ", " << itB->y << "\n";
        itB++;
    }



    float2* cudaBoids;
    gpuErrchk(cudaMalloc(&cudaBoids, numBoids * sizeof(float2)));
    gpuErrchk(cudaMemcpy(cudaBoids, localBoids, numBoids*sizeof(float2), cudaMemcpyHostToDevice));


    // Allocate result of transformation in device memory
    float2* output;
    gpuErrchk(cudaMalloc(&output, numBoids * sizeof(float2)));

    bool rule2 = true;
    if(!rule2) {
        // Invoke kernel
        dim3 dimBlock(256);
        dim3 dimGrid((numBoids + dimBlock.x - 1) / dimBlock.x);
        applyBoidsVelocityRuleKernel<<<dimGrid, dimBlock>>>(cudaBoids, output,
                                                            cuArray, numBoids, width, height, FIXED_RANGE);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

    }else{
        float2 *boids_vel = (float2*) malloc(width*height * sizeof(float2));

        for (Boid& boid : boids){
            auto p = boid.getPosition();
//        cout << "**have point : " << p.x << ", "<<  p.y << "\n";
            auto vec = boid.getVelocity();
            auto vel = vec.getDisplacement();
            float2 velf = {vel.x, vel.y};
            if(p.x >= 0 && p.y >=0 && p.x < width && p.y < height)
                boids_vel[(size_t)p.y*width+(size_t)p.x] = velf;
//        cout << "--wrote point in kernel: " << itB->x<< ", " << itB->y << "\n";
            
        }
        
        float2 * cuda_boids_vel;
        gpuErrchk(cudaMalloc(&cuda_boids_vel, width*height*sizeof(float2)));
        gpuErrchk(cudaMemcpy(cuda_boids_vel, boids_vel, width*height*sizeof(float2), cudaMemcpyHostToDevice));

        free(boids_vel);

        dim3 dimBlock(256);
        dim3 dimGrid((numBoids + dimBlock.x - 1) / dimBlock.x);
        applyBoidsVelocityAndColisionRulesKernel<<<dimGrid, dimBlock>>>(cudaBoids, output, numBoids,
                                                          cuArray, cuda_boids_vel, width, height, FIXED_RANGE, COLLISION_RANGE);

//        dim3 dimBlock(32, 32);
//        size_t square = floor(sqrt(numBoids)) + 1;
//        dim3 dimGrid((square + dimBlock.x - 1) / dimBlock.x,
//                     (square + dimBlock.y - 1) / dimBlock.y);
        applyBoidsVelocityAndColisionRulesKernel<<<dimGrid, dimBlock>>>(cudaBoids, output, numBoids,
                                                                        cuArray, cuda_boids_vel, width, height,
                                                                        FIXED_RANGE, COLLISION_RANGE);



        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaFree(cuda_boids_vel));

    }

//    cudaMemcpy(h_wind, output, size,cudaMemcpyDeviceToHost);
    gpuErrchk(cudaMemcpy(localBoids, output, numBoids*sizeof(float2), cudaMemcpyDeviceToHost));
    // Destroy texture object
//    gpuErrchk(cudaDestroyTextureObject(texObj));
    // Free device memory
    gpuErrchk(cudaFree(cuArray));
    gpuErrchk(cudaFree(cudaBoids));
    gpuErrchk(cudaFree(output));

    itB = localBoids;
    cout << "Updating boids location based on GPU output\n";
    for(Boid& boid: boids){
        std::vector<Point2f> points;
        float2 val = *itB;
        Point2f targetPoint = Point2f(val.x, val.y);
        points.push_back(targetPoint);
        if(DEBUG) {
            if (abs(val.x - boid.getPosition().x) > 0.5 || abs(val.y - boid.getPosition().y) > 0.5) {
                cout << "Boid " << (itB - localBoids) << " (" << boid.getPosition().x << "," << boid.getPosition().y
                     << ") will update position with (dx, dy) = (" << val.x << "," << val.y << ")\n";

            } else {
                cout << "Not updating Boid " << (itB - localBoids) << " (" << boid.getPosition().x << ","
                     << boid.getPosition().y << ")\t with result (" << val.x << "," << val.y << ")\n";
            Point2f wind = windMap.at<Point2f>((int)boid.getPosition().y, (int)boid.getPosition().x);

            int x = boid.getPosition().x;
            int y = boid.getPosition().y;

            for(int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    wind = windMap.at<Point2f>(y + i, x + j);
                    cout << "windmap at boid position (" << x + j << ", " << y + i << "): " << wind.x << "," << wind.y
                         << ")\n";
                }
            }
//            cout<< "Boid not updated";
            }
        }

        boid.updateVelocity(points);
        itB++;
    }
//    Mat out = windMap;


    free(h_wind);
    free(localBoids);

//    drawScene();
    return true;
}

/**
 * Updates the simulation by applying rules to each boid's motion using CUDA.
 * These rules are combined to create target points towards each boid object will navigate.
 *
 * @return true after completion (beta)...
 */
bool CUDAScene::updateSimulation2D() {

    //boids are given as a 1D float2 array and the Vectors are given as a 2D float4 texture
    //each thread processes one boid and computes its new velocity and stores it in
    //a 1D float2 array with each element on the original poisition.

//    Mat& windMap = this->windMap;
    int width = windMap.cols;
    int height = windMap.rows;

    cout << "Preparing GPU data...\n";

//    // Allocate CUDA array in device memory
//    cudaChannelFormatDesc channelDesc =
//            cudaCreateChannelDesc(32, 32, 0, 0,
//                                  cudaChannelFormatKindFloat);

    // Copy to device memory some data located at address h_wind
    // in host memory
    size_t size = height*width;
    float2* cuArray;
    gpuErrchk(cudaMalloc(&cuArray, size*sizeof(float2)));

    float2 *h_wind = (float2*) malloc(size * sizeof(float2));
    float2* it = h_wind;
    for(int i = 0; i < windMap.rows; i++){
        for(int j = 0; j < windMap.cols; j++){
            auto p = windMap.at<Point2f>(i, j);
//            cout << "**have point : " << p.x << ", "<<  p.y << "\n";
            *it = {(float)p.x, (float)p.y};
//            cout << "--wrote point in kernel: " << it->x<< ", " << it->y << "\n";
            it++;
        }
    }


    gpuErrchk(cudaMemcpy(cuArray, h_wind, size*sizeof(float2), cudaMemcpyHostToDevice));

    // Allocate the boids arrays
    size_t numBoids = boids.size();
    size_t matLen = trunc(sqrt(numBoids)) + 1;
    float2* localBoids = (float2*) malloc(matLen * matLen * sizeof(float2));
//    float2* itB = localBoids;

    float2 *boids_vel = (float2*) malloc(width*height * sizeof(float2));

    float *counts = (float*) malloc(width*height*sizeof(float));
    memset(boids_vel, 0, width*height*sizeof(float2));
    memset(counts, 0, width*height*sizeof(float));

    int i;
    float max = -9999;
    float min = 9999;
    float maxMag = 0;
    for (i = 0; i < boids.size(); i++) {
        Boid &boid = boids[i];
        const Point2f& p = boid.getPosition();
//        cout << "**have point : " << p.x << ", "<<  p.y << "\n";
        localBoids[i] = {(float)p.x, (float)p.y};

        auto vec = boid.getVelocity();
        auto vel = vec.getDisplacement();
        float2 velf = {vel.x, vel.y};
        int x = (int)std::round(p.x);
        int y = (int)std::round(p.y);
        if(x >= 0 && y >=0 && x < width && y < height) {

            float mag = sqrt(velf.x*velf.x+velf.y*velf.y);
            if(mag > 0) {
//                boids_vel[y * width + x].x += mag * velf.x;
//                boids_vel[y * width + x].y += mag * velf.y;
//                counts[y * width + x] += mag;
            boids_vel[y * width + x].x += velf.x;
            boids_vel[y * width + x].y += velf.y;
            counts[y*width+x] +=1;
                if (max < velf.x)
                    max = velf.x;
                if (min > velf.x)
                    min = velf.x;
                if(maxMag < mag)
                    maxMag = mag;
            }
        }
//        cout << "--wrote point in kernel: " << itB->x<< ", " << itB->y << "\n";

    }
    std::cout << " max boid velocity: " << max << endl;
    std::cout << " min boid velocity: " << min << endl;
    std::cout << " max boid magnitude: " << maxMag << endl;

//    for(i = 0; i < width*height; i++){
//        if(counts[i] > 1){
////            i % 100 && cout << "have velocity " << boids_vel[i].x << "," << boids_vel[i].y << "\n";
//            boids_vel[i].x /= counts[i];
//            boids_vel[i].y /= counts[i];
////            i % 100 && cout << "set velocity " << boids_vel[i].x << "," << boids_vel[i].y << "\n";
//        }
//    }

    float2 * cuda_boids_vel;
    float *cuda_boids_mag;
    gpuErrchk(cudaMalloc(&cuda_boids_vel, width*height*sizeof(float2)));
    gpuErrchk(cudaMalloc(&cuda_boids_mag, width*height*sizeof(float)));
    gpuErrchk(cudaMemcpy(cuda_boids_vel, boids_vel, width*height*sizeof(float2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cuda_boids_mag, counts, width*height*sizeof(float), cudaMemcpyHostToDevice));


    float2* cudaBoids;
    gpuErrchk(cudaMalloc(&cudaBoids, matLen * matLen * sizeof(float2)));
    gpuErrchk(cudaMemcpy(cudaBoids, localBoids, matLen * matLen*sizeof(float2), cudaMemcpyHostToDevice));


    // Allocate result of transformation in device memory
    float2* output;
    gpuErrchk(cudaMalloc(&output, matLen * matLen * sizeof(float2)));
    gpuErrchk(cudaMemset(output, 0, matLen*matLen*sizeof(float2)));

    bool rule2 = true;
    bool particleOnly = false;
    if(!rule2) {
        // Invoke kernel
        dim3 dimBlock(256);
        dim3 dimGrid((numBoids + dimBlock.x - 1) / dimBlock.x);
        applyBoidsVelocityRuleKernel<<<dimGrid, dimBlock>>>(cudaBoids, output,
                                                            cuArray, numBoids, width, height, FIXED_RANGE);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

    }else{

//        dim3 dimBlock(16, 16);
//        dim3 dimGrid((numBoids + dimBlock.x - 1) / dimBlock.x);
        dim3 dimBlock(16, 16);
        dim3 dimGrid((matLen + dimBlock.x - 1) / dimBlock.x,
                     (matLen + dimBlock.y - 1) / dimBlock.y);
        if(particleOnly){
            applyScaledBoidsVelocityAndColisionRulesKernel2D<<<dimGrid, dimBlock>>>(cudaBoids, output, numBoids, matLen,
                                                                                cuArray, cuda_boids_vel, cuda_boids_mag, width, height,
                                                                                     0, COLLISION_RANGE);
        }else {
            applyScaledBoidsVelocityAndColisionRulesKernel2D<<<dimGrid, dimBlock>>>(cudaBoids, output, numBoids, matLen,
                                                                                    cuArray, cuda_boids_vel,
                                                                                    cuda_boids_mag, width, height,
                                                                                    FIXED_RANGE, COLLISION_RANGE);
        }


        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaFree(cuda_boids_vel));
        gpuErrchk(cudaFree(cuda_boids_mag));

    }

    gpuErrchk(cudaMemcpy(localBoids, output, matLen * matLen*sizeof(float2), cudaMemcpyDeviceToHost));
    // Destroy texture object
//    gpuErrchk(cudaDestroyTextureObject(texObj));
    // Free device memory

//    itB = localBoids;
    cout << "Updating boids location based on GPU output\n";

//#pragma omp parallel private(i)
    {
//#pragma omp single
//        cout << "Updating on " << omp_get_num_threads() << " threads\n";
//#pragma omp for
    size_t m = 0;
    for (i = 0; i < numBoids; i++) {
        Boid &boid = boids[i];
        std::vector<Point2f> points;
        float2 val = localBoids[i];
        Point2f neighbourTarget = Point2f(val.x, val.y);



        if(neighbourTarget.x != 0 && neighbourTarget.y != 0) {

            Point2f pos = boid.getPosition();
//            if (abs(pos.x - val.x) > 100 || abs(pos.y - val.y) > 100) {
//                cout << "boid position: " << boid.getPosition() << endl;
//                cout << "target position: " << neighbourTarget << endl;
//                cout << "boid velocity: " << boid.getVelocity().getDisplacement() << endl;
//            }
            points.push_back(neighbourTarget);
//
//
            boid.updateVelocity(points);
        }else{
            Point2f singleTarget = boid.getPosition();
            singleTarget.x += boid.getVelocity().getDisplacement().x;
            singleTarget.y += boid.getVelocity().getDisplacement().y;

            points.push_back(singleTarget);

            boid.updateVelocity(points);

            m++;
        }

//
//        i%100 && cout << "updated position: " << boid.getPosition() << endl;
//        i%100 && cout << "updated velocity: " << boid.getVelocity().getDisplacement() << endl;
//        itB++;
    }
        cout << "moved to 0,0: " << m << endl;
    }

//    Mat out = windMap;



    gpuErrchk(cudaFree(cuArray));
    gpuErrchk(cudaFree(cudaBoids));
    gpuErrchk(cudaFree(output));

    free(boids_vel);
    free(counts);

    free(h_wind);
    free(localBoids);

    drawScene();
    return true;
}

/**
 * yeah... clears the scene using Mat::zeros
 */
void CUDAScene::clearScene() {
    scene = Mat::zeros( sizeY, sizeX, CV_8UC3 );
}

/**
 * Starts the simulation by drawing the very first frame.
 *
 * @return true after completion (beta)...
 */
bool CUDAScene::startSimulation(string outFolder, int startIndex, int radius) {
    this->framesSaved = startIndex;
    this->outputFolder = outFolder;
    FIXED_RANGE = (float) radius;
    saveSimulation = true;
//    namedWindow("OpticalFlow", WINDOW_AUTOSIZE);
    drawScene();
    return true;
}

/**
 * Clears the scene of previous drawings, then redraws it using the updated boid positions.
 * After saving the scene to an image on disk, it will display the scene and wait for user key press to continue;
 */
void CUDAScene::drawScene() {
    clearScene();
    for (Boid boid : boids) {

        const cv::Point point = cv::Point(cvRound(boid.getPosition().x), cvRound(boid.getPosition().y));
        if(point.x < 0 || point.x > 1919 || point.y < 0 || point.y > 1079)
            continue;
//        cout << "Drawing circle at point: (" << point.x << "," << point.y << ")\n";
        circle(scene, point, .5, cv::Scalar(255, 255, 255, 0), cv::FILLED);
    }


//    FileHelper::writeFile("/Users/mariuspenteliuc/Assets/PhD/debug/debug_out/boids/boids_" + std::string(5 - to_string(framesSaved).length(), '0') + std::to_string(framesSaved) + ".jpg", scene);
    FileHelper::writeFile(outputFolder + "/simulated_clouds/boids_" + std::string(5 - to_string(framesSaved).length(), '0') + std::to_string(framesSaved) + ".jpg", scene);
    framesSaved++;
//    imshow("OpticalFlow", scene);
//    if (CUDAScene::previewSimulation) {
//        std::cout << "press any key to continue..." << std::endl;
//        waitKey();
//    }
}

void CUDAScene::drawCloudMask(const Mat& mask, float threshold, int maxBoids) {
    clearScene();

    for(int i = 0; i < 1080; i++){
        for(int j = 0; j < 1920; j++){
            auto point = mask.at<Point2f>(i, j);
            if(point.x >= threshold || point.y >= threshold){
                int x = (int)point.x;
                circle(scene, Point2f(j, i), .5f, cv::Scalar(128, 128, 128+x, 0), cv::FILLED);
            }
        }
    }

//    FileHelper::writeFile("/Users/mariuspenteliuc/Assets/PhD/debug/debug_out/boids/boids_" + std::string(5 - to_string(framesSaved).length(), '0') + std::to_string(framesSaved) + ".jpg", scene);
    FileHelper::writeFile(outputFolder + "/simulated_clouds/mask_" + std::string(5 - to_string(framesSaved).length(), '0') + std::to_string(framesSaved) + ".jpg", scene);
//    framesSaved++;
//    imshow("OpticalFlow", scene);
//    if (CUDAScene::previewSimulation) {
//        std::cout << "press any key to continue..." << std::endl;
//        waitKey();
//    }
}

void CUDAScene::drawCloudMaskDifference(const Mat &predM, const Mat &realM, float threshold, int realThreshold) {
    clearScene();

    for(int i = 0; i < 1080; i++){
        for(int j = 0; j < 1920; j++){
            auto real = realM.at<Vec3b>(i, j);
            auto pred = predM.at<Point2f>(i, j);
            bool realMask = abs(real[0]) > realThreshold || abs(real[1]) > realThreshold;
            bool predMask = abs(pred.x) >= threshold || abs(pred.y) >= threshold;

            if(realMask){
                if(!predMask){
                    //false negative
                    circle(scene, Point2f(j, i), .5f, cv::Scalar(128, 255, 255, 0), cv::FILLED);
                }else{
                    //true positive
                    circle(scene, Point2f(j, i), .5f, cv::Scalar(128, 255, 128, 0), cv::FILLED);
                }
            }else{
                if(!predMask){
                    //true negative
//                    circle(scene, Point2f(j, i), .5f, cv::Scalar(255, 128, 255, 0), cv::FILLED);
                }else{
                    //false positive
                    circle(scene, Point2f(j, i), .5f, cv::Scalar(255, 128, 255, 0), cv::FILLED);
                }
            }


        }
    }

//    FileHelper::writeFile("/Users/mariuspenteliuc/Assets/PhD/debug/debug_out/boids/boids_" + std::string(5 - to_string(framesSaved).length(), '0') + std::to_string(framesSaved) + ".jpg", scene);
    FileHelper::writeFile(outputFolder + "/simulated_clouds/mask_diff" + std::string(5 - to_string(framesSaved).length(), '0') + std::to_string(framesSaved) + ".jpg", scene);
//    framesSaved++;
//    imshow("OpticalFlow", scene);
//    if (CUDAScene::previewSimulation) {
//        std::cout << "press any key to continue..." << std::endl;
//        waitKey();
//    }
}

/// Computes the difference between two wind maps and outputs that to another wind map using CUDA.
///
/// The more similar two wind maps are, the lower displacement values will the resulted map have and vice versa.
/// @param first one wind map being the left hand side term.
/// @param second another wind map being the right hand side term.
/// @param result the object where the values will be saved.
bool CUDAScene::computeDifferenceOfWindMaps(const Mat& first, const Mat& second, Mat& result) {


    int width = first.cols;
    int height = first.rows;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 32, 0, 0,
                                  cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    gpuErrchk(cudaMallocArray(&cuArray, &channelDesc, width, height));
    // Copy to device memory some data located at address h_wind
    // in host memory
    size_t size = height*width;
    float2 *h_wind = (float2*) malloc(size * sizeof(float2));
    float2* it = h_wind;
    for(int i = 0; i < first.rows; i++){
        for(int j = 0; j < first.cols; j++){
            auto p = first.at<Point2f>(i, j);
            *it = {p.x, p.y};
            it++;
        }
    }


    gpuErrchk(cudaMemcpyToArray(cuArray, 0, 0, h_wind, size*sizeof(float2), cudaMemcpyHostToDevice));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    gpuErrchk(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    float2* secondCuda;
    gpuErrchk(cudaMalloc(&secondCuda, width * height * sizeof(float2)));

    gpuErrchk(cudaMemcpy(secondCuda, h_wind, size,cudaMemcpyHostToDevice));



    // Allocate result of transformation in device memory
    float2* output;
    gpuErrchk(cudaMalloc(&output, width * height * sizeof(float2)));


    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    matrixDifferenceUsingTexturesKernel<<<dimGrid, dimBlock>>>(secondCuda, output,
                                                         texObj, width, height);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_wind, output, size,cudaMemcpyDeviceToHost));
    // Destroy texture object
    gpuErrchk(cudaDestroyTextureObject(texObj));
    // Free device memory
    gpuErrchk(cudaFreeArray(cuArray));
    gpuErrchk(cudaFree(output));
    gpuErrchk(cudaFree(secondCuda));


//    Mat out = input;

    it = h_wind;
    for(int i = 0; i < result.rows; i++){
        for(int j = 0; j < result.cols; j++){
            result.at<Point2f>(i,j).x = it->x;
            result.at<Point2f>(i,j).y = it->y;
            it++;
        }
    }

    free(h_wind);

    return true;
}

/**
 * Runs the simulation by updating the scene a specified number of times.
 *
 * @param steps an integer representing how many times should the simulation run.
 * @return true after completion (beta)...
 */
bool CUDAScene::runSimulation(int steps, bool preview) {
    this->previewSimulation = preview;
    for (int i = 0; i < steps; ++i) {
//        updateSimulation();
        updateSimulation2D();
    }
    return true;
}
