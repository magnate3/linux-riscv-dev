cmake_minimum_required(VERSION 3.18)
project(paged_attn LANGUAGES C CUDA)

# ───────── options ─────────
option(PA_BUILD_TESTS "Build test executables" ON)
set(PA_CUDA_ARCH "52;61" CACHE STRING "CUDA architectures to target (sm_52=Maxwell, sm_61=Pascal)")

# ───────── library ─────────
add_library(paged_attn STATIC
    paged_attn.cu
    kv_pager.c
)

target_include_directories(paged_attn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(paged_attn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${PA_CUDA_ARCH}"
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 11
)

# Mixed f16/f32 requires sm_53+ for __half operations, but we gate
# on sm_52 and use __half2float/__float2half which are available on all arches.
target_compile_options(paged_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
    >
    $<$<COMPILE_LANGUAGE:C>:
        -Wall -Wextra -O2
    >
)

find_package(Threads REQUIRED)
target_link_libraries(paged_attn PRIVATE Threads::Threads)

# ───────── test ─────────
if(PA_BUILD_TESTS)
    add_executable(test_paged_attn test_paged_attn.cu)
    target_link_libraries(test_paged_attn PRIVATE paged_attn)
    set_target_properties(test_paged_attn PROPERTIES
        CUDA_ARCHITECTURES "${PA_CUDA_ARCH}"
    )
endif()
