//
//  main.cu
//  CloudMovement-CUDA
//
//  Created by Adrian Spătaru on 26.02.2021.
//

#include "kernels.hpp"

using namespace std;


int main(int argc, const char * argv[]) {

//    createCloudMaskFromFlow();
//    createCloudMasks();
//        OpticalFlowService ofService;
//    OpticalFlowService::computeFlowForImages("../images", "../images", "png", false, true, false);

    if(argc < 4){
        cerr << "Not enough arguments!";
        programOptions(argv[0]);
        exit(0);
    }else{
        argParse(argc, argv);
    }
    srand (static_cast <unsigned> (time(0)));



//    for (int configuration = 0; configuration < 12; ++configuration) {
//        cout << "Running experiment " << configuration << endl;
//        runExperiments(configuration);
//    }



    runExperiments();
//    computeMAPE();



//    testCuda();
//    simpletestCuda();


    return 0;
}

