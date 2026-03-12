cmake_minimum_required(VERSION 3.21)
project(kv-compact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KV_COMPACT_BUILD_TESTS "Build unit tests" ON)
option(KV_COMPACT_BUILD_TOOL  "Build CLI tool (requires llama.cpp)" ON)

# ---- Header-only math library (no dependencies) ----
add_library(kv-compact-math INTERFACE)
target_include_directories(kv-compact-math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Unit tests (standalone, no llama.cpp needed) ----
if(KV_COMPACT_BUILD_TESTS)
    add_executable(test-kv-compact-math tests/test-kv-compact-math.cpp)
    target_link_libraries(test-kv-compact-math PRIVATE kv-compact-math)
endif()

# ---- CLI tool (needs llama.cpp) ----
if(KV_COMPACT_BUILD_TOOL)
    # Find llama.cpp - either via LLAMA_CPP_DIR or FetchContent
    if(LLAMA_CPP_DIR)
        # Use a local llama.cpp checkout
        message(STATUS "Using local llama.cpp: ${LLAMA_CPP_DIR}")
        set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
        add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
    else()
        # Fetch from GitHub
        include(FetchContent)
        FetchContent_Declare(
            llama_cpp
            GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
            GIT_TAG        master
            GIT_SHALLOW    TRUE
        )
        message(STATUS "Fetching llama.cpp from GitHub (this may take a while)...")
        FetchContent_MakeAvailable(llama_cpp)
    endif()

    add_executable(llama-kv-compact src/kv-compact.cpp)
    target_link_libraries(llama-kv-compact PRIVATE
        kv-compact-math
        llama
        common
        ggml
    )

    # E2E test (also needs llama.cpp)
    add_executable(test-kv-compact-e2e tests/test-kv-compact-e2e.cpp)
    target_link_libraries(test-kv-compact-e2e PRIVATE
        kv-compact-math
        llama
        common
        ggml
    )
endif()
