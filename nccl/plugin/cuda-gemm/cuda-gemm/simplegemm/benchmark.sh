#!/bin/bash

export TORCHINDUCTOR_CACHE_DIR=/tmp/pingpong_matmul_experiments_20250310
export TORCHINDUCTOR_CUTLASS_DIR=$HOME/local/cutlass
export TORCHINDUCTOR_CUTLASS_ALLOWLIST='128x128x64_1x1x1.*pingpong_epi_tma'
export TORCHINDUCTOR_CUTLASS_DENYLIST='stream_k'
export TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL=0201
export USE_IR_LOC=ttgir

DATE=$(date +%s)
export TRITON_DUMP_DIR=$(realpath "dump.$DATE")

RUN_COMMAND="python benchmark.py"

if false; then
    export TRITON_OVERRIDE_DIR=$(realpath "override.$DATE")

    echo $TRITON_DUMP_DIR
    echo $TRITON_OVERRIDE_DIR

    rm -rf $TRITON_DUMP_DIR $TRITON_OVERRIDE_DIR

    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 ./denoise-h100.sh $RUN_COMMAND
    cp -r $TRITON_DUMP_DIR $TRITON_OVERRIDE_DIR
    TTGIR_PATH=$(find $TRITON_OVERRIDE_DIR -name 'matmul_persistent_tma_ws_pingpong_kernel.ttgir')
    find $TRITON_OVERRIDE_DIR -type f -delete
    cp matmul_persistent_tma_ws_pingpong_kernel.ttgir $TTGIR_PATH
fi

export BENCHMARK_CUTLASS=1
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 ./denoise-h100.sh $RUN_COMMAND
