cmake_minimum_required(VERSION 3.24)
project(triton_detection_postprocessing_backend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA not found. Please install CUDA.")
endif()
message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
enable_language(CUDA)

include(FetchContent)
FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG "r25.12"
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG "r25.12"
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG "r25.12"
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

add_library(
  triton_postprocess SHARED
  src/main.cu
)

target_link_libraries(
  triton_postprocess
  PRIVATE
    triton-core-serverapi
    triton-core-backendapi
    triton-core-serverstub
    triton-backend-utils
)

set_target_properties(
  triton_postprocess PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
