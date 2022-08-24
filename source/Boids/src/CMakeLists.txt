include_directories(${cuda_movement_SOURCE_DIR}/Boids/include)
include_directories(${cuda_movement_SOURCE_DIR}/Helpers/include)
add_library(Boids Boid.cpp Scene.cpp CUDAScene.cu Vector.cpp)
set_property(TARGET Boids PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)