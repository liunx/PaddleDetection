#!/bin/bash

#PADDLE_DIR=$(realpath InferLibrary/inference_lite_lib.x86)
PADDLE_DIR=$(realpath InferLibrary/inference_lite_lib.arm)

rm -rf build
mkdir -p build
cd build

cmake .. \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DPADDLE_LIB_NAME=libpaddle_light_api_shared

make -j8
