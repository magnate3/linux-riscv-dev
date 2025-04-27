cmake_minimum_required(VERSION 3.10)
project(rdma)

include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -Wall -Wextra -Wpedantic")

link_libraries(ibverbs)
file(GLOB SOURCES src/*.c ./common/*.c)

add_executable (rc_pingpong ${SOURCES} rc_pingpong.c)
add_executable (ud_pingpong ${SOURCES} ud_pingpong.c)
