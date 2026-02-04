cmake_minimum_required(VERSION 3.10) 
 
project(evaluate) 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) 
set(CMAKE_CXX_FLAGS -g) 

aux_source_directory(src/ SOURCE_DIR)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

if(USE_SYSTEM_NCNN)
    find_package(ncnn)
    if(NOT TARGET ncnn)
        message(WARNING "ncnn target not found! USE_SYSTEM_NCNN will be turned off.")
        set(USE_SYSTEM_NCNN OFF)
    endif()
endif()

if(NOT USE_SYSTEM_NCNN)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! Please update submodules with \"git submodule update --init --recursive\" and try again.")
endif()
### Options
    option(NCNN_BUILD_BENCHMARK "build benchmark" OFF)
    add_subdirectory(ncnn)
endif()

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_DIR}) 
 
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ncnn) 
target_link_libraries(${CMAKE_PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ncnn)
#target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC ${EXTRA_LIBS})
