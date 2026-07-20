#!/bin/bash


filename=$(basename "$1")
name="${filename%.*}"

cmake $SDE/p4studio/ -DCMAKE_INSTALL_PREFIX=$SDE/install/ -DCMAKE_MODULE_PATH=$SDE/cmake/ -DP4_NAME=${name} -DP4_PATH=$1
make ${name}
make install
