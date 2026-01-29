#!/bin/bash


set -e
PROJECT_ROOT="$(cd "$(dirname $0)/.." && pwd)"
NCNN_DIR="${PROJECT_ROOT}/lib/ncnn"
BUILD_DIR="${NCNN_DIR}/build"
INSTALL_DIR="${BUILD_DIR}/install"

echo "=========================================="
echo "Compile NCNN"
echo "=========================================="

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ${NCNN_DIR} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DNCNN_SHARED_LIB=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_VULKAN=OFF \
    -DNCNN_OPENMP=ON \
    -DNCNN_THREADS=ON \
    -DNCNN_RUNTIME_CPU=ON \
    -DNCNN_AVX2=OFF \
    -DNCNN_AVX=OFF \
    -DNCNN_SSE2=OFF \
    -DNCNN_RVV=OFF \
    -DNCNN_VFPV4=OFF \
    -DNCNN_ARM82=OFF \
    -DNCNN_DISABLE_RTTI=OFF \
    -DNCNN_DISABLE_EXCEPTION=OFF \
    -DNCNN_INT8=ON \
    -DNCNN_BF16=OFF \
    -DNCNN_PIXEL=ON \
    -DNCNN_PIXEL_ROTATE=ON \
    -DNCNN_PIXEL_AFFINE=ON \
    -DNCNN_PIXEL_DRAWING=OFF \
    -DCMAKE_CXX_FLAGS="-D__riscv_vector=0 -fopenmp"

make -j$(nproc)
make install

echo "=========================================="
echo "NCNN Compilation done！"
echo "Install PATH: ${INSTALL_DIR}"
echo "=========================================="