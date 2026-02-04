cmake_minimum_required(VERSION 2.8.12...3.10)
project(ncnn_benchmark)

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (MSVC)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif ()

# 添加一个外部传入的ncnn库路径
option(NCNN_DIR "Path to ncnn library" "")
message(STATUS "NCNN_DIR: ${NCNN_DIR}")

# 查找ncnn库
list(APPEND CMAKE_PREFIX_PATH ${NCNN_DIR})
find_package(ncnn REQUIRED)
include_directories(${NCNN_DIR}/include)
link_directories(${NCNN_DIR}/lib)
if (CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "debug")
    add_definitions(-DDEBUG_MODE)
    link_libraries(ncnnd)
else ()
    link_libraries(ncnn)
endif ()

add_definitions(-DNOMINMAX)

add_executable(benchncnn benchncnn.cpp)
