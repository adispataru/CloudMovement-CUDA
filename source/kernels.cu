#include "kernels.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


std::string OUTPUT_FOLDER;

std::string FLOW_FOLDER;
std::string MASK_FOLDER;

std::string INPUT_MAP;

int maxBoids = 1000;

int STEPS  = 2;

int BOIDS_PER_CLOUD = 10;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ void transformToSimpleDirectionCUDA(float2 *p){
    float number = p->x;
    if (number < 0.5 && number > -0.5) {
        number = 0;
    } else if (number >= 0.5) {
        number = 1;
    } else {
	number = -1;
    }
    p->x = number;

    number = p->y;
    if (number < 0.5 && number > -0.5) {
        number = 0;
    } else if (number >= 0.5) {
        number = 1;
    } else {
        number = -1;
    }

    p->y = number;
}

__global__ void computeMAPEKernel(float2* realWindMap, float2* predictedWindMap,
                                  float* mapeDirectionOut, float2* mapeOut,
                                  float percent, size_t width, size_t height){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > width-(width*percent) || y > height-(height*percent))
        return;
    if(x < width*percent || y < height*percent)
        return;

    float2 real = realWindMap[y*width+x];
    float2 pred = predictedWindMap[y*width+x];

    size_t numCols = width-2*round(width*percent);
    size_t numRows = height-2*round(height*percent);
    size_t numElems = numCols * numRows;
//    float2 forecastPoint = transformToSimpleDirection(pred);
//    float2 realPoint = transformToSimpleDirection(real);
//
//    real.x = round(real.x) - trunc(real.x); does not work for 2.3 -> 0
    float2 realPoint = real;
    float2 forecastPoint = pred;
    transformToSimpleDirectionCUDA(&realPoint);
    transformToSimpleDirectionCUDA(&forecastPoint);
    float mape_error_1p = 0.0f;
    if ((realPoint.x > 0 || realPoint.y > 0) && (realPoint.x != forecastPoint.x || realPoint.y != forecastPoint.y) && realPoint.x - forecastPoint.x == 0 && realPoint.y - forecastPoint.y == 0) {
        mape_error_1p += 100.0 / numElems;
    } else {
//        float distance = Vector::getEuclidianDistance(forecastPoint, realPoint);
        float distance = sqrt(pow(forecastPoint.x - realPoint.x, 2) + pow(forecastPoint.y - realPoint.y, 2));
        if (distance == 0) {

        } else if (distance <= 1) {
            mape_error_1p += 25.0 / numElems;
        } else if (distance <= 2) {
           mape_error_1p += 50.0 / numElems;
        } else mape_error_1p += 75.0 / numElems;
    }

//    Point2f point = computeAPE(real, forecastPoint);
//    float2 point = abs(real.x - pred.x) / real.x)*100, (abs(real.y - forecast.y) / real.y)*100);
    float2 point = {(abs(real.x - pred.x) / abs(real.x))*100, (abs(real.y - pred.y) / abs(real.y))*100};
    float mape_x = point.x / (1.0f*numElems);
    float mape_y = point.y / (1.0f*numElems);
    mapeOut[y*width+x] = {mape_x, mape_y};
    mapeDirectionOut[y*width+x] = mape_error_1p;
}

__global__ void computeMAEUsingScalarProductKernel(float2* realWindMap, float2* predictedWindMap,
                                  double* mapeDirectionOut,
                                  float percent, size_t width, size_t height){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > width-(width*percent) || y > height-(height*percent))
        return;
    if(x < width*percent || y < height*percent)
        return;

    float2 real = realWindMap[y*width+x];
    float2 pred = predictedWindMap[y*width+x];

    size_t numCols = width-2*round(width*percent);
    size_t numRows = height-2*round(height*percent);
    size_t numElems = numCols * numRows;
//    float2 forecastPoint = transformToSimpleDirection(pred);
//    float2 realPoint = transformToSimpleDirection(real);
//
//    real.x = round(real.x) - trunc(real.x); does not work for 2.3 -> 0
//    float2 realPoint = real;
//    float2 forecastPoint = pred;
    double nom = (real.x * pred.x + real.y * pred.y); //
//            (sqrt(pow(real.x, 2) + pow(real.y, 2))
//            * sqrt(pow(pred.x, 2) + pow(pred.y, 2)));

    double denom = (sqrt(pow(real.x, 2) + pow(real.y, 2))
            * sqrt(pow(pred.x, 2) + pow(pred.y, 2)));

    double angle = 0;
    if(denom != 0) {
        angle = nom/denom;
    }
    if(abs(angle) <= 1.0 )
       angle = acos(angle);
    else{
	angle = 0;
    }
//    float mae_error_angle = (angle / M_PI) * numElems;
    double mae_error_angle = angle / M_PI;
    mapeDirectionOut[y*width+x] = mae_error_angle / (1.0 * numElems);
//    Point2f point = computeAPE(real, forecastPoint);
//    float2 point = abs(real.x - pred.x) / real.x)*100, (abs(real.y - forecast.y) / real.y)*100);
//    float2 point = {(abs(real.x - pred.x) / abs(real.x))*100, (abs(real.y - pred.y) / abs(real.y))*100};
//    float mape_x = point.x / (1.0f*numElems);
//    float mape_y = point.y / (1.0f*numElems);

}


__global__ void computeMapeUsingScalarProductKernel(float2* realWindMap, float2* predictedWindMap,
                                                    double* mapeDirectionOut,
                                                    float percent, size_t width, size_t height){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > width-(width*percent) || y > height-(height*percent))
        return;
    if(x < width*percent || y < height*percent)
        return;

    float2 real = realWindMap[y*width+x];
    float2 pred = predictedWindMap[y*width+x];

    size_t numCols = width-2*round(width*percent);
    size_t numRows = height-2*round(height*percent);
    size_t numElems = numCols * numRows;

    float2 ref = {0, -1};
    double realNom = (real.x * ref.x + real.y * ref.y);

    double predNom = (pred.x * ref.x + pred.y * ref.y);

    double realDenom = (sqrt(pow(real.x, 2) + pow(real.y, 2))
                    * sqrt(pow(ref.x, 2) + pow(ref.y, 2)));

    double predDenom = (sqrt(pow(pred.x, 2) + pow(pred.y, 2))
                         * sqrt(pow(ref.x, 2) + pow(ref.y, 2)));
    double realAngle = 0;
    if(realDenom != 0) {
        realAngle = realNom/realDenom;
    }
    if(abs(realAngle) <= 1.0 )
        realAngle = acos(realAngle);
    else{
	realAngle = 0;
    }

    double predAngle = 0;
    if(realDenom != 0) {
        predAngle = predNom/predDenom;
    }
    if(abs(predAngle) <= 1.0 )
        predAngle = acos(predAngle);
    else{
        predAngle = 0;
    }
//    predAngle /= M_PI;
//    realAngle /= M_PI;
    predAngle = predAngle*180/M_PI;
    realAngle = realAngle*180/M_PI;
//    float mape_error_angle = (angle / M_PI) * numElems;
    if(realAngle != 0 && predAngle != 0){
        double mape_error_angle = abs((realAngle - predAngle)/realAngle);
        mapeDirectionOut[y*width+x] = mape_error_angle/(1.0*numElems);
    }



}


__global__ void computeCloudMaskAccuracyKernel(float2* realWindMap, float2* predictedWindMap, char4* results,
                                                    float percent, size_t width, size_t height, float threshold){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > width-(width*percent) || y > height-(height*percent))
        return;
    if(x < width*percent || y < height*percent)
        return;

    float2 real = realWindMap[y*width+x];
    float2 pred = predictedWindMap[y*width+x];

    size_t numCols = width-2*round(width*percent);
    size_t numRows = height-2*round(height*percent);
//    size_t numElems = numCols * numRows;
//    float2 forecastPoint = transformToSimpleDirection(pred);
//    float2 realPoint = transformToSimpleDirection(real);
//
//    real.x = round(real.x) - trunc(real.x); does not work for 2.3 -> 0

    bool realMask = abs(real.x) > 0 || abs(real.y) > 0;
    bool predMask = abs(pred.x) >= threshold || abs(pred.y) >= threshold;

    if(realMask){
        if(!predMask){
            //false negative
            results[y*width+x].w = 1;
        }else{
            //true positive
            results[y*width+x].x = 1;
        }
    }else{
        if(!predMask){
            //true negative
            results[y*width+x].y = 1;
        }else{
            //false positive
	    results[y*width+x].z = 1;
        }
    }
}

using namespace std;

void argParse(int argc, const char **argv) {
    FLOW_FOLDER = string(argv[1]);
    MASK_FOLDER = string(argv[2]);
    OUTPUT_FOLDER = string(argv[3]);

    int i = 4;
    if(argc > i){
        maxBoids = atoi(argv[i]);
        i++;
    }
    if(argc > i){
        STEPS = atoi(argv[i]);
        i++;
    }
    if(argc > i){
        INPUT_MAP = string(argv[i]);
        i++;
    }

}

void programOptions(const char *string) {
    cout << string << "wind-folder mask-folder output-folder [num-boids steps map-image]\n";

}


void countTrueAndFalsePositivesAndNegativesOnGPU(Mat forecastedWindMap, Mat realWindMap, size_t *TP, size_t *TN, size_t *FP,
                                                 size_t *FN, float threshold) {


    unsigned long rows = realWindMap.rows;
    unsigned long int cols = realWindMap.cols;
    unsigned long matAllocSize = rows*cols * sizeof(float2);

//    cout << "Allocating real wind map\n";
    float2 *realWind = (float2*) malloc(matAllocSize);
    float2 *it = realWind;
    for(int i = 0; i< realWindMap.rows; i++){
        for(int j = 0; j < realWindMap.cols; j++){
            auto p = realWindMap.at<Point2f>(i, j);
            *it = {p.x, p.y};
            it++;
        }
    }

//    cout << "Allocating predicted wind map\n";
    float2 *predWind = (float2*) malloc(matAllocSize);
    it = predWind;
    for(int i = 0; i< forecastedWindMap.rows; i++){
        for(int j = 0; j < forecastedWindMap.cols; j++){
            auto p = forecastedWindMap.at<Point2f>(i, j);
            *it = {p.x, p.y};
            it++;
        }
    }
    //cuda input arrays: real wind map and predicted wind map
    float2 *realWindCuda, *predWindCuda;
    gpuErrchk(cudaMalloc(&realWindCuda, matAllocSize));
    gpuErrchk(cudaMalloc(&predWindCuda, matAllocSize));
    gpuErrchk(cudaMemcpy(realWindCuda, realWind, matAllocSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(predWindCuda, predWind, matAllocSize, cudaMemcpyHostToDevice));

    char4 *cudaResults;
    gpuErrchk(cudaMalloc(&cudaResults, rows*cols*sizeof(char4)));


    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
                 (rows + dimBlock.y - 1) / dimBlock.y);

//    cout << "Allocating true positives results matrix\n";
    char4 *results = (char4 *) malloc(rows*cols*sizeof(char4));

    int x = 0;
    for(float percentage : {0.0f, 0.1f, 0.2f, 0.4f}) {
        gpuErrchk(cudaMemset(cudaResults, 0, rows*cols*sizeof(char4)));
        computeCloudMaskAccuracyKernel<<<dimGrid, dimBlock>>>(realWindCuda, predWindCuda, cudaResults, percentage, cols, rows, threshold);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(results, cudaResults, rows*cols*sizeof(char4), cudaMemcpyDeviceToHost));

        char4* rit = results;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                TP[x] += rit->x;
                TN[x] += rit->y;
                FP[x] += rit->z;
                FN[x] += rit->w;
                rit++;
            }


        }

        x++;
    }

    gpuErrchk(cudaFree(cudaResults));
    gpuErrchk(cudaFree(realWindCuda));
    gpuErrchk(cudaFree(predWindCuda));
    free(results);
    free(realWind);
    free(predWind);

}


void countTrueAndFalsePositivesAndNegativesOnGPUMask(Mat forecastedWindMap, Mat realWindMap, size_t *TP, size_t *TN, size_t *FP,
                                                 size_t *FN, float threshold) {


    unsigned long rows = realWindMap.rows;
    unsigned long int cols = realWindMap.cols;
    unsigned long matAllocSize = rows*cols * sizeof(float2);

//    cout << "Allocating real wind map\n";
    float2 *realWind = (float2*) malloc(matAllocSize);
    float2 *it = realWind;
    for(int i = 0; i< realWindMap.rows; i++){
        for(int j = 0; j < realWindMap.cols; j++){
            auto p = realWindMap.at<Point2f>(i, j);
            *it = {p.x, p.y};
            it++;
        }
    }

//    cout << "Allocating predicted wind map\n";
    float2 *predWind = (float2*) malloc(matAllocSize);
    it = predWind;
    for(int i = 0; i< forecastedWindMap.rows; i++){
        for(int j = 0; j < forecastedWindMap.cols; j++){
            auto p = forecastedWindMap.at<Point2f>(i, j);
            *it = {p.x, p.y};
            it++;
        }
    }
    //cuda input arrays: real wind map and predicted wind map
    float2 *realWindCuda, *predWindCuda;
    gpuErrchk(cudaMalloc(&realWindCuda, matAllocSize));
    gpuErrchk(cudaMalloc(&predWindCuda, matAllocSize));
    gpuErrchk(cudaMemcpy(realWindCuda, realWind, matAllocSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(predWindCuda, predWind, matAllocSize, cudaMemcpyHostToDevice));

    char4 *cudaResults;
    gpuErrchk(cudaMalloc(&cudaResults, rows*cols*sizeof(char4)));


    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
                 (rows + dimBlock.y - 1) / dimBlock.y);

//    cout << "Allocating true positives results matrix\n";
    char4 *results = (char4 *) malloc(rows*cols*sizeof(char4));

    int x = 0;
    for(float percentage : {0.0f, 0.1f, 0.2f, 0.4f}) {
        gpuErrchk(cudaMemset(cudaResults, 0, rows*cols*sizeof(char4)));
        computeCloudMaskAccuracyKernel<<<dimGrid, dimBlock>>>(realWindCuda, predWindCuda, cudaResults, percentage, cols, rows, threshold);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(results, cudaResults, rows*cols*sizeof(char4), cudaMemcpyDeviceToHost));

        char4* rit = results;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                TP[x] += rit->x;
                TN[x] += rit->y;
                FP[x] += rit->z;
                FN[x] += rit->w;
                rit++;
            }


        }

        x++;
    }

    gpuErrchk(cudaFree(cudaResults));
    gpuErrchk(cudaFree(realWindCuda));
    gpuErrchk(cudaFree(predWindCuda));
    free(results);
    free(realWind);
    free(predWind);

}

void computeMapeAndMAEOnGPU(Mat &forecastedWindMap, Mat &realWindMap, double *mape_error_p, double *mae_error_p) {
    unsigned long rows = realWindMap.rows;
    unsigned long int cols = realWindMap.cols;
    unsigned long matAllocSize = rows*cols * sizeof(float2);

    float2 *realWind = (float2*) malloc(matAllocSize);
    float2 *it = realWind;
    for(int i = 0; i< realWindMap.rows; i++){
        for(int j = 0; j < realWindMap.cols; j++){
            auto p = realWindMap.at<Point2f>(i, j);
            *it = {p.x, p.y};
            it++;
        }
    }

    float2 *predWind = (float2*) malloc(matAllocSize);
    it = predWind;
    for(int i = 0; i< forecastedWindMap.rows; i++){
        for(int j = 0; j < forecastedWindMap.cols; j++){
            auto p = forecastedWindMap.at<Point2f>(i, j);
            *it = {p.x, p.y};
            it++;
        }
    }
    //cuda input arrays: real wind map and predicted wind map
    float2 *realWindCuda, *predWindCuda;
    gpuErrchk(cudaMalloc(&realWindCuda, matAllocSize));
    gpuErrchk(cudaMalloc(&predWindCuda, matAllocSize));
    gpuErrchk(cudaMemcpy(realWindCuda, realWind, matAllocSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(predWindCuda, predWind, matAllocSize, cudaMemcpyHostToDevice));

    //cuda output arrays: direction mape  (float), total mape (float2);

    double *dirMapeCuda;

    gpuErrchk(cudaMalloc(&dirMapeCuda, rows * cols * sizeof(double)));

    //host output arrays
    double *dirMape = (double *) malloc(rows*cols*sizeof(double));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
                 (rows + dimBlock.y - 1) / dimBlock.y);

    int x = 0;
    for(float percentage : {0.0f, 0.1f, 0.2f, 0.4f}) {

        gpuErrchk(cudaMemset(dirMapeCuda, 0, rows * cols * sizeof(double)));

        computeMapeUsingScalarProductKernel<<<dimGrid, dimBlock>>>(realWindCuda, predWindCuda, dirMapeCuda,
                                                                   percentage, cols, rows);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(dirMape, dirMapeCuda, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i =  rows*percentage; i < rows-(rows*percentage); i++){
            for(int j = cols*percentage; j < cols-(cols*percentage); j++){

                mape_error_p[x] += dirMape[i*cols+j];

            }
        }

        gpuErrchk(cudaMemset(dirMapeCuda, 0, rows * cols * sizeof(double)));

        computeMAEUsingScalarProductKernel<<<dimGrid, dimBlock>>>(realWindCuda, predWindCuda, dirMapeCuda,
                                                                  percentage, cols, rows);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(dirMape, dirMapeCuda, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));
        for(int i =  rows*percentage; i < rows-(rows*percentage); i++){
            for(int j = cols*percentage; j < cols-(cols*percentage); j++){

                mae_error_p[x] += dirMape[i*cols+j];

            }
        }

        x++;
    }

    gpuErrchk(cudaFree(dirMapeCuda));
    gpuErrchk(cudaFree(realWindCuda));
    gpuErrchk(cudaFree(predWindCuda));

    free(dirMape);
    free(realWind);
    free(predWind);
}


void computeMAPE(){

    vector<size_t> startIndex = {0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0};
    vector<int> numberOfFlows = {0, 48, 48, 48, 336, 336, 336, 336, 1460, 1460, 1460, 1460 };
    vector<int> neighborhoodRadius = {5, 5, 11, 31, 0, 5, 11, 31, 0, 5, 11, 31, 0, 5, 11, 31 };
    vector<String> outputFolders;

    for(int i = 0; i < 12; i++){
        outputFolders.push_back(OUTPUT_FOLDER + to_string(i));
    }

    string SEQ_FOLDER="/mnt/disks/data/parallel-boids-bucket/sequential-experiment/simulated-wind-maps";

    cout << "Computing MAPE for  " << STEPS << " steps with:\nFLOW_FODLER: "
         << FLOW_FOLDER << "\nSEQUENTIAL_FOLDER: " << SEQ_FOLDER << "\n";

    vector<cv::String> fileNames, seqFileNames;
    glob(FLOW_FOLDER + "/*.npy", fileNames, false);
    Mat realFlow = FileHelper::readFile(fileNames[0]);

    glob(SEQ_FOLDER + "/*.npy", seqFileNames, false);
    Mat seqFlow = FileHelper::readFile(seqFileNames[0]);


    ofstream ofs2;
    ofs2.open (OUTPUT_FOLDER+ "/MAPE_Conf_seq.csv", ofstream::out | ofstream::app);


    ofs2 << "%STEP,MAE,MAE(trim10),MAE(trim20),MAE(trim40),MAPE,MAPE(trim10),MAPE(trim20),MAPE(trim40),SENSITIVITY,SPECIFICITY,PRECISION\n";

    for (int k = 1; k < STEPS; ++k) { // to keep things fast
//            scene.runSimulation(1);

        Mat realFlow = FileHelper::readFile(fileNames[k]);
        Mat seqFlow = FileHelper::readFile(seqFileNames[k]);

        double mae_error_p[4] = {0,0,0,0};
        double mape_error_p[4] = {0,0,0,0};

        computeMapeAndMAEOnGPU(seqFlow, realFlow, mape_error_p, mae_error_p);

        size_t TP[] = {0,0,0,0}, TN[] = {0,0,0,0}, FP[] = {0,0,0,0}, FN[] = {0,0,0,0};
        countTrueAndFalsePositivesAndNegativesOnGPU(seqFlow, realFlow, TP, TN, FP, FN, 1.0f);

        //compute accuracy functions
        float specificity[] = {0,0,0,0};
        float sensitivity[] = {0,0,0,0};
        float precision[] = {0,0,0,0};
        for(int i = 0; i < 4; i++){
            //probability of predicting a cloud given that there is indeed a cloud there
            sensitivity[i] = (float)TP[i] / (float)(TP[i] + FN[i]);
            //probability of predicting NO cloud given that there is indeed no cloud there
            specificity[i] = (float) TN[i] / (float)(TN[i] + FP[i]);

            precision[i] = (float) TP[i] / (float)(TP[i] + FP[i]);
        }


        ofs2 <<  k << "," << mae_error_p[0] << "," << mae_error_p[1] << "," << mae_error_p[2] << "," <<
             mae_error_p[3] << "," << mape_error_p[0] << "," << mape_error_p[1] << "," << mape_error_p[2] << "," <<
             mape_error_p[3] << "," << sensitivity[0] <<  "," << specificity[0] <<  "," << precision[0] << endl;


//            ofs << startIndex[CONFIG] + k << " — " <<  " MAPE: <" << mape_error << "> 10% trim: <" << mape_error_1p << "> 20% trim: <" << mape_error_2p << "> 40% trim: <" << mape_error_4p << "> 10% edge: <" << mape_error_1p_out <<  "> 20% edge: <" << mape_error_2p_out <<  "> 40% edge: <" << mape_error_4p_out << ">\t// " << fileNames[startIndex[CONFIG] + k] << " AND " << outputFolders[CONFIG] << "/simulated_wind_map/wind_map_" << string(5 - to_string(startIndex[CONFIG] + k).length(), '0') << to_string(startIndex[CONFIG] + k) << ".jpg" << "— Time:" << elapsed_seconds.count() << endl;
//            ofs2 << startIndex[CONFIG] + k << " — " <<  " MAPE: <" << mape_error << "> 10% trim: <" << mape_error_1p << "> 20% trim: <" << mape_error_2p << "> 40% trim: <" << mape_error_4p << "> 10% edge: <" << mape_error_1p_out <<  "> 20% edge: <" << mape_error_2p_out <<  "> 40% edge: <" << mape_error_4p_out << ">\t// " << fileNames[startIndex[CONFIG] + k] << " AND " << outputFolders[CONFIG] << "/simulated_wind_map/wind_map_" << string(5 - to_string(startIndex[CONFIG] + k).length(), '0') << to_string(startIndex[CONFIG] + k) << ".jpg" << "— Time:" << elapsed_seconds.count() << endl;
//        ofs << startIndex[CONFIG] + k << "," << mape_error << "," << fileNames[startIndex[CONFIG] + k] << "," << outputFolders[CONFIG] << "/simulated_wind_map/wind_map_" << string(5 - to_string(startIndex[CONFIG] + k).length(), '0') << to_string(startIndex[CONFIG] + k) << ".jpg" << "," << elapsed_seconds.count() << endl;
    }
    ofs2 << endl;
    ofs2.close();
//    }


}


void createCloudMasks() {
    vector<cv::String> maskNames;
    glob("/home/adrian/CLionProjects/cuda-movement/images/*.jpg", maskNames, false);

    vector<cv::String> flows;
    glob("/home/adrian/CLionProjects/cuda-movement/images/flows/*.npy", flows, false);


    int i = 0;
    Mat prevMat;
    for(i = 1; i < flows.size(); i++){
        string filename = maskNames[i];
        Mat img = FileHelper::readFile(filename);
        Mat imgLAB, imgHSV, imgYCB;

        cvtColor(img, imgLAB, COLOR_BGR2Lab);
        cvtColor(img, imgHSV, COLOR_BGR2HSV);
        cvtColor(img, imgYCB, COLOR_BGR2YCrCb);

        if(img.empty())
            cerr << "error reading jpg file";

        Mat m2, result;
        m2.copySize(img);
        result.copySize(img);

//        cv::cvtColor(img, m2, cv::COLOR_BGR2GRAY);
//        cv::threshold(m2, result, 150, 200, 0);

        Vec3b bgrPixel(160, 160, 160);
        Mat3b bgr (bgrPixel);

//        int thresh = 40;

        Scalar minBGR = Scalar(90, 90, 90);
        Scalar maxBGR = Scalar(250, 250, 250);
//

        Mat frame_HSV, frame_threshold;

        //process image to check if it has the trimmed section
        //if it does, we replace it with pixels from previous frame
        for(int y = 0; y < img.rows; y++){
            for(int x = 0;  x < img.cols; x++){
                auto p = img.at<Vec3b>(y, x);
                if(p[0] <= 10 && p[1] <= 10 && p[2] <= 10){
//                    cout << "negru\n";
                    img.at<Vec3b>(y, x) = prevMat.at<Vec3b>(y, x);
//                    cout << img.at<Vec3b>(y, x) << "\n";
                }
                if(p[2] > p[1]+10 && p[2] > p[0]+10){
                    img.at<Vec3b>(y, x) = {55,89,78};
                }
            }
        }
        cv::cvtColor(img, imgHSV, COLOR_BGR2HSV);
        Mat hsv[3];
        cv::split(imgHSV, hsv);


        Mat img_sat;
        bitwise_not(hsv[1], img_sat);
//        bitwise_not(imgHSV, imgHSV);
//        cv::threshold(img_sat, result, 180, 255, THRESH_BINARY);


        ///values not chosen arbitrarily
        //any hue is accepted white pixels have hue 179, black pixels have 0
        //saturation is key. 0 means all pixels have same value (ie, shadow of pure grey)
        //      but we accept any saturation value up to 20%, meaning the diff between max and min channel is 20%.
        //      this allows to detect clouds over land or sea.
        //      20% transalates to roughlly 37.5 on a scale from 0..255
        inRange(imgHSV, Scalar(0, 0, 70), Scalar(140, 20, 255), result);
//
//        Mat maskBGR, flowMask;
//        inRange(img, minBGR, maxBGR, maskBGR);
//        bitwise_and(img, img, flowMask, maskBGR);

//        cv::cvtColor(flowMask, result, cv::COLOR_BGR2GRAY);

        Mat flow = FileHelper::readFile(flows[i-1]);

        for(int ii = 0; ii < flow.rows; ii++){
            for(int j = 0; j < flow.cols; j++){
                auto p = flow.at<Point2f>(ii, j);
                if(abs(p.x) < 1 && abs(p.y) < 1){
                    result.at<unsigned char>(ii, j) = 0;
                }
            }
        }

        img.copyTo(prevMat);

        char s[512] = {0};
        sprintf(s, "%s/mask_%05d.png", flows[i-1].substr(0, flows[i-1].find_last_of('/')).c_str(), i);
        cout << "writing " << s << endl;
//        FileHelper::writeFile(filename.substr(0, filename.find_last_of("/")) + ".png", m2);
        FileHelper::writeFile(s, result);


    }
}

void createCloudMaskFromFlow() {
    vector<cv::String> flows;
    glob("/home/adrian/CLionProjects/cuda-movement/images/flows/*.npy", flows, false);

    int x = 0;
    for(auto filename : flows){
        Mat flow = FileHelper::readFile(filename);

        Mat resultBGR = Mat::zeros(flow.rows, flow.cols, CV_8U);

        for(int i = 0; i < flow.rows; i++){
            for(int j = 0; j < flow.cols; j++){
                auto p = flow.at<Point2f>(i, j);
                if(abs(p.x) >= 1 && abs(p.y) >= 1){
                    resultBGR.at<unsigned char>(i,j) = (p.x*p.x + p.y*p.y)*10;
                }
            }
        }


        char s[512];
        sprintf(s, "%s/mask-flow_%05d.png", filename.substr(0, filename.find_last_of("/")).c_str(), x);
        cout << "writing " << s << endl;
        FileHelper::writeFile( s, resultBGR);




        x++;
    }
}



int runExperiments(int CONFIG) {

    cout << "Starting simulation of " << STEPS << " steps with:\nBOIDS:  " << maxBoids << "\nFLOW_FODLER: "
         << FLOW_FOLDER << "\nMASK_FOLDER: " << MASK_FOLDER << "\nOUTPUT_FOLDER: " << OUTPUT_FOLDER << "\n";

    vector<size_t> startIndex = {1, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0};
    vector<int> numberOfFlows = {0, 48, 48, 48, 336, 336, 336, 336, 1460, 1460, 1460, 1460 };
    vector<int> neighborhoodRadius = {5, 5, 11, 31, 0, 5, 11, 31, 0, 5, 11, 31, 0, 5, 11, 31 };
    vector<String> outputFolders;

    for(int i = 0; i < 12; i++){
        outputFolders.push_back(OUTPUT_FOLDER + to_string(i));
    }

    vector<cv::String> fileNames, maskNames;
    glob(FLOW_FOLDER + "/*.npy", fileNames, false);
    glob(MASK_FOLDER + "/*.png", maskNames, false);

    Mat flow = FileHelper::readFile(fileNames[startIndex[CONFIG]]);


    cout << "Using " << numberOfFlows[CONFIG] << " averaged flows." << endl;

    CUDAScene scene = CUDAScene(1920, 1080);
    scene.updateWindMap(flow);
    ;

    Mat initialMask = FileHelper::readFile(maskNames[startIndex[CONFIG]]);
    Mat checked = Mat::zeros(initialMask.rows, initialMask.cols, CV_8U);
    if(maxBoids != 0) {
        while (scene.getBoidsCount() < maxBoids) {
            int row = rand() % initialMask.rows;
            int col = rand() % initialMask.cols;
            if (checked.at<bool>(row, col))
                continue;
            checked.at<bool>(row, col) = true;
            auto fxy = initialMask.at<Point2f>(row, col);
            if (abs(fxy.x) >= 1 || abs(fxy.y) >= 1) {

//            if (rand() % 2) {
                scene.addBoid(col, row, 0, fxy.x, fxy.y);
//            }

            }
        }
    }else{
        for(int i = 0; i < initialMask.rows; i++){
            for (int j = 0; j < initialMask.cols; j++){
                auto cloud = initialMask.at<Vec3b>(i, j);
                auto fxy = flow.at<Point2f>(i, j);
                if (cloud[0] + cloud[1] + cloud[2] > 10) {
                    int x = 0;
                    while(x++ < BOIDS_PER_CLOUD /*&& x < max(fxy.x, fxy.y)*/) {
                        scene.addBoid(j, i, 2, fxy.x, fxy.y);
                    }

                }
            }
        }
    }


    scene.startSimulation(outputFolders[CONFIG], startIndex[CONFIG], neighborhoodRadius[CONFIG]);
    ofstream ofs2;
    ofs2.open (OUTPUT_FOLDER+ "/MAPE_Conf_" + to_string(CONFIG) + ".csv", ofstream::out | ofstream::app);

    ofs2 << "%Configuration " << CONFIG << ": " << endl;
    ofs2 << "%STEP,MAPE,MAPE(trim10),MAPE(trim20),MAPE(trim40),MAE,MAE(trim10),MAE(trim20),MAE(trim40),RECALL,SPECIFICITY,PRECISION,F1-SCORE,RECALL(boids),SPECIFICITY(boids),PRECISION(boids),F1-SCORE(boids),filename,time(boids),time(windmap)\n";

    for (int k = 1; k < STEPS; ++k) { // to keep things fast
//            scene.runSimulation(1);

        cout << "Step [" << k << "] with " << scene.getBoidsCount() << " boids: \n";
        auto startBoids = std::chrono::system_clock::now();
        scene.runSimulation(1);
        auto endBoids = std::chrono::system_clock::now();

        auto start = std::chrono::system_clock::now();
//        scene.updateWindMapUsingBoids(neighborhoodRadius[CONFIG]);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = end-start;
        std::chrono::duration<double> elapsed_secondsBoids = endBoids-startBoids;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);

        Mat forecastedWindMap = scene.getWindMap();
        Mat realWindMap = FileHelper::readFile(fileNames[numberOfFlows[CONFIG] + k]);

        cout << "TEST OUT: " << forecastedWindMap.empty() << endl;

        double mae_error_p[4] = {0,0,0,0};
        double mape_error_p[4] = {0,0,0,0};

        computeMapeAndMAEOnGPU(forecastedWindMap, realWindMap, mape_error_p, mae_error_p);

        size_t TP[] = {0,0,0,0}, TN[] = {0,0,0,0}, FP[] = {0,0,0,0}, FN[] = {0,0,0,0};

        Mat mask = FileHelper::readFile(maskNames[startIndex[CONFIG] + k]);
        countTrueAndFalsePositivesAndNegativesOnGPU(forecastedWindMap, realWindMap, TP, TN, FP, FN, 1.0f);
//        countTrueAndFalsePositivesAndNegativesOnGPU(forecastedWindMap, mask, TP, TN, FP, FN, 1.0f);

        //compute accuracy functions
        float specificity[] = {0,0,0,0};
        float recall[] = {0, 0, 0, 0};
        float precision[] = {0,0,0,0};
        float f1_score[] = {0,0,0,0};
        for(int i = 0; i < 4; i++){
            //probability of predicting a cloud given that there is indeed a cloud there
            recall[i] = (float)TP[i] / (float)(TP[i] + FN[i]);
            //probability of predicting NO cloud given that there is indeed no cloud there
            specificity[i] = (float) TN[i] / (float)(TN[i] + FP[i]);

            precision[i] = (float) TP[i] / (float)(TP[i] + FP[i]);

            f1_score[i] = 2 / ((1/recall[i]) + (1/precision[i]));
        }

        size_t TPB[] = {0,0,0,0}, TNB[] = {0,0,0,0}, FPB[] = {0,0,0,0}, FNB[] = {0,0,0,0};
        //reevaluate forecast based on boids
        cout << "Computing cloud mask based on boids\n";
        Mat boidsMap = Mat::zeros(forecastedWindMap.rows, forecastedWindMap.cols, CV_32FC2);
        float max = 0;
        size_t inside = 0, outside = 0;
        for(auto boid : scene.getAllBoids()){
            auto pos = boid.getPosition();
            int x = (int) floor(pos.x);
            int y = (int) floor(pos.y);
            if(x > 0 && x < boidsMap.cols && y > 0 && y < boidsMap.rows) {
                inside++;
                boidsMap.at<Point2f>(y, x).x += 1.0f;
                if (boidsMap.at<Point2f>(y, x).x > max) {
                    max = boidsMap.at<Point2f>(y, x).x;
//                    cout << "max (" << max << ") at " << pos << endl;
                }
                boidsMap.at<Point2f>(y, x).y += 1;
            }else{
                outside++;
            }
        }
        cout << "Constructed cloud mask with boids: [in/out] = [" << inside << "/" << outside << "]\n";
        inside = 0;
        outside = 0;
        for(int i = 0; i < boidsMap.rows; i++){
            for(int j = 0; j < boidsMap.cols; j++){
                auto p =  boidsMap.at<Point2f>(i, j);
                if(p.x > 0)
                    inside+=1;
                else
                    outside+=1;
            }
        }
        cout << "Drawing cloud mask with clouds: [clouds/empty] = [" << inside << "/" << outside << "]\n";

        scene.drawCloudMask(boidsMap, (float)BOIDS_PER_CLOUD / 2, (int)max);
        scene.drawCloudMaskDifference(boidsMap, mask, (float)BOIDS_PER_CLOUD / 2, 0);



        cout << "Max boids per pixel: " << max << endl;
        cout << "Counting true positives and stuff.. \n";
        countTrueAndFalsePositivesAndNegativesOnGPUMask(boidsMap, mask, TPB, TNB, FPB, FNB, (float) BOIDS_PER_CLOUD / 2);

        //compute accuracy functions
        float specificityBoids[] = {0,0,0,0};
        float recallBoids[] = {0, 0, 0, 0};
        float precisionBoids[] = {0,0,0,0};
        float f1_scoreBoids[] = {0,0,0,0};
        for(int i = 0; i < 4; i++){
            //probability of predicting a cloud given that there is indeed a cloud there
            recallBoids[i] = (float)TPB[i] / (float)(TPB[i] + FNB[i]);
            //probability of predicting NO cloud given that there is indeed no cloud there
            specificityBoids[i] = (float) TNB[i] / (float)(TNB[i] + FPB[i]);

            precisionBoids[i] = (float) TPB[i] / (float)(TPB[i] + FPB[i]);

            f1_scoreBoids[i] = 2 / ((1/precisionBoids[i]) + (1/recallBoids[i]));
        }

        cout << "Boids TP: " << TPB[0] << "\tTN: " << TNB[0] << "\tFP: " << FPB[0] << "\tFN: " << FNB[0];

        ofs2 << startIndex[CONFIG] + k << "," << mape_error_p[0] << "," << mape_error_p[1] << "," << mape_error_p[2] << "," <<
             mape_error_p[3] << "," << mae_error_p[0] << "," << mae_error_p[1] << "," << mae_error_p[2] << "," <<
             mae_error_p[3] << "," << recall[0] << "," << specificity[0] << "," << precision[0] << "," << f1_score[0] << "," <<
             recallBoids[0] << "," << specificityBoids[0] << "," << precisionBoids[0] << "," << f1_scoreBoids[0] << ","
             << fileNames[startIndex[CONFIG] + k] << "," << elapsed_secondsBoids.count()<<","
             << elapsed_seconds.count() << endl;

        cout << "boids elapsed time"<< elapsed_secondsBoids.count() << "\twind map elapsed time: " << elapsed_seconds.count() << "s\n";

//            ofs << startIndex[CONFIG] + k << " — " <<  " MAPE: <" << mape_error << "> 10% trim: <" << mape_error_1p << "> 20% trim: <" << mape_error_2p << "> 40% trim: <" << mape_error_4p << "> 10% edge: <" << mape_error_1p_out <<  "> 20% edge: <" << mape_error_2p_out <<  "> 40% edge: <" << mape_error_4p_out << ">\t// " << fileNames[startIndex[CONFIG] + k] << " AND " << outputFolders[CONFIG] << "/simulated_wind_map/wind_map_" << string(5 - to_string(startIndex[CONFIG] + k).length(), '0') << to_string(startIndex[CONFIG] + k) << ".jpg" << "— Time:" << elapsed_seconds.count() << endl;
//            ofs2 << startIndex[CONFIG] + k << " — " <<  " MAPE: <" << mape_error << "> 10% trim: <" << mape_error_1p << "> 20% trim: <" << mape_error_2p << "> 40% trim: <" << mape_error_4p << "> 10% edge: <" << mape_error_1p_out <<  "> 20% edge: <" << mape_error_2p_out <<  "> 40% edge: <" << mape_error_4p_out << ">\t// " << fileNames[startIndex[CONFIG] + k] << " AND " << outputFolders[CONFIG] << "/simulated_wind_map/wind_map_" << string(5 - to_string(startIndex[CONFIG] + k).length(), '0') << to_string(startIndex[CONFIG] + k) << ".jpg" << "— Time:" << elapsed_seconds.count() << endl;
//        ofs << startIndex[CONFIG] + k << "," << mape_error << "," << fileNames[startIndex[CONFIG] + k] << "," << outputFolders[CONFIG] << "/simulated_wind_map/wind_map_" << string(5 - to_string(startIndex[CONFIG] + k).length(), '0') << to_string(startIndex[CONFIG] + k) << ".jpg" << "," << elapsed_seconds.count() << endl;
    }
    ofs2 << endl;
    ofs2.close();
//    }

    return 0;
}
