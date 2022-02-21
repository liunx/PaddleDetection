#!/bin/bash

rm -rf build
mkdir -p build
cd build

#PADDLE_DIR=${HOME}/Work/Robot/PaddlePaddle/InferLibrary/inference_lite_lib.with_log
PADDLE_DIR=/work/PaddlePaddle/InferLibrary/paddlelite_inference

cmake .. \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DPADDLE_LIB_NAME=libpaddle_light_api_shared

make -j8
