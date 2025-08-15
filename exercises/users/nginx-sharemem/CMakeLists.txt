project(sharemem)
#cmake_minimum_required(VERSION 3.10.2)
cmake_minimum_required(VERSION 2.8)

aux_source_directory(. SRC_LIST)
aux_source_directory(./src SRC_LIST)
SET(CMAKE_BUILD_TYPE Debug)
# 显示编译过程
set(CMAKE_VERBOSE_MAKEFILE ON)

set(EXE_OBJ sharemem_test)

INCLUDE_DIRECTORIES(./)
INCLUDE_DIRECTORIES(./src)
add_definitions("-Wall -ggdb3 -O0")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib-test)
ADD_LIBRARY(${PROJECT_NAME} SHARED ${SRC_LIST})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} pthread)

link_directories(${PROJECT_SOURCE_DIR}/lib-test)


add_executable(${EXE_OBJ} ${PROJECT_SOURCE_DIR}/test/test.c)
TARGET_LINK_LIBRARIES(${EXE_OBJ} ${PROJECT_NAME})

add_executable(tree ${PROJECT_SOURCE_DIR}/test/tree.c)
TARGET_LINK_LIBRARIES(tree ${PROJECT_NAME})
