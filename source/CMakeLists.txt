cmake_minimum_required(VERSION 3.14)
project(cuda_movement CXX CUDA)

#set(CUDAFLAGS -Xcompiler="-mno-float128")
#set(CUDA_HOST_COMPILER "/mnt/hpc-storage/homes/adispataru/anaconda3/envs/cuda-dev/bin/g++")
#set(CMAKE_CUDA_COMPILER "/mnt/hpc-storage/homes/adispataru/anaconda3/envs/cuda-dev/bin/nvcc")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_ARCHITECTURES 70)
#set_property(TARGET cuda_movement PROPERTY CUDA_ARCHITECTURES 70)
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

set(CMAKE_CUDA14_STANDARD_COMPILE_OPTION "-std=c++11")
set(CMAKE_CUDA14_EXTENSION_COMPILE_OPTION "-std=c++11")

#set (CUDA_NVCC_FLAGS "-ccbin=${CXX}")

link_libraries(stdc++fs)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP REQUIRED)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}; -Xcompiler \"${OpenMP_CXX_FLAGS}\"")

#   set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} -lgomp")
#endif()



add_subdirectory(Boids)
add_subdirectory(Helpers)
include_directories(${cuda_movement_SOURCE_DIR}/Boids/include)
include_directories(${cuda_movement_SOURCE_DIR}/Helpers/include)
link_directories(${cuda_movement_SOURCE_DIR}/Boids/src)
link_directories(${cuda_movement_SOURCE_DIR}/Helpers/src)

target_link_libraries(Helpers ${OpenCV_LIBS})
target_link_libraries(Boids Helpers)


#add_executable(cuda_movement main.cu)
add_executable(cuda_movement main2.cpp kernels.hpp kernels.cu)
target_link_libraries( cuda_movement Helpers Boids stdc++fs ${OpenCV_LIBS} )
#target_link_libraries( cuda_movement stdc++fs )
#target_link_libraries(cuda_movement Helpers)
#target_link_libraries(cuda_movement Boids)
#target_link_libraries(Boids stdc++fs)
#target_link_libraries(Boids OpenMP::OpenMP_CXX)
#target_link_libraries(Helpers stdc++fs)
#target_link_libraries(cuda_movement OpenMP::OpenMP_CXX)


link_libraries(Boids)
link_libraries(Helpers)


set_property(TARGET Boids PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cuda_movement PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
