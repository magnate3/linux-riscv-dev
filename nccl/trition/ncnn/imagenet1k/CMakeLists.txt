cmake_minimum_required(VERSION 2.8)

set(ncnn_root_dir /pytorch/ncnn)
set(OpenCV_INC_DIR /usr/local/include/opencv4)
set(OpenCV_LIB_DIR /usr/local/lib/)
project(image_classifier)

#set(CMAKE_CXX_STANDARD 17)
add_definitions(-std=c++17)
add_definitions("-Wall")
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#include_directories(${ncnn_root_dir}/build/install/include/ncnn)    # change ncnn_root_dir to yours
include_directories(${OpenCV_INC_DIR} /usr/include/jsoncpp ${ncnn_root_dir}/build/install/include/ncnn)    # change ncnn_root_dir to yours
#link_directories(${ncnn_root_dir}/build/install/lib)             # change ncnn_root_dir to yours
link_directories(${OpenCV_LIB_DIR}/lib/ ${ncnn_root_dir}/build/install/lib)             # change ncnn_root_dir to yours

set(OpenCV_LIBS opencv_core 
                opencv_highgui 
                opencv_imgproc 
                opencv_imgcodecs 
                opencv_videoio)

add_executable(image_classifier image_classifier.cpp)
add_executable(resnet resnet.cpp)
add_executable(batch_resnet batch_resnet.cpp LoadData.cpp)
target_link_libraries(image_classifier ncnn ${OpenCV_LIBS})
target_link_libraries(resnet  ncnn ${OpenCV_LIBS} jsoncpp)
target_link_libraries(batch_resnet  ncnn ${OpenCV_LIBS} jsoncpp)
