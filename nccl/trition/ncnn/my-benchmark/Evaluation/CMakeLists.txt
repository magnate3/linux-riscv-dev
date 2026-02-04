cmake_minimum_required(VERSION 2.8.12...3.10)
 
project(evaluate) 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) 
set(CMAKE_CXX_FLAGS -g) 
aux_source_directory(src/ SOURCE_DIR)

find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (MSVC)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif ()

# 添加一个外部传入的ncnn库路径
option(NCNN_SRC_DIR "Path to ncnn src library" "")
option(NCNN_INSTALL_DIR "Path to ncnn install library" "")
message(STATUS "NCNN_INSTALL_DIR: ${NCNN_INSTALL_DIR}")
message(STATUS "NCNN_SRC_DIR: ${NCNN_SRC_DIR}")

# 查找ncnn库
list(APPEND CMAKE_PREFIX_PATH ${NCNN_INSTALL_DIR})
find_package(ncnn REQUIRED)
include_directories(${NCNN_INSTALL_DIR}/include)
include_directories(${NCNN_SRC_DIR})
link_directories(${NCNN_INSTALL_DIR}/lib)
if (CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "debug")
    add_definitions(-DDEBUG_MODE)
    link_libraries(ncnnd)
else ()
    link_libraries(ncnn)
endif ()

add_definitions(-DNOMINMAX)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_DIR})

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ncnn)
target_link_libraries(${CMAKE_PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ncnn)
