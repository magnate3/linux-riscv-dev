
#
# simple-backend

set(CMAKE_VERBOSE_MAKEFILEON ON)
set(GGML_ROOT /pytorch/GGML-Tutorial/ggml)
set(LIB_GGML_DIR /pytorch/GGML-Tutorial/build/ggml/src/)
#set(TEST_TARGET simple-sgemm)
set(TEST_TARGET2 test)
# first step
include_directories(${GGML_ROOT}/include) 
# secnod step
link_directories(${LIB_GGML_DIR})
MESSAGE( STATUS "LIB_GGML_DIR = ${LIB_GGML_DIR}")
# 3th step
#add_executable(${TEST_TARGET} simple-sgemm.cpp)
add_executable(${TEST_TARGET2} main.cpp)
# 4th step
#target_link_libraries(${TEST_TARGET} ggml ggml-cpu ggml-base)
target_link_libraries(${TEST_TARGET2} ggml ggml-cpu ggml-base)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
