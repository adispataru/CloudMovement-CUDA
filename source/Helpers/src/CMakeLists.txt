include_directories(${cuda_movement_SOURCE_DIR}/Helpers/include)
add_library(Helpers FileHelper.cpp MathHelper.cpp OpticalFlowService.cpp)