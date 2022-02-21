#!/bin/bash

MODEL=$1
IMG=$2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:Libs
./build/main \
    -use_mkldnn \
    -run_benchmark \
    -cpu_threads 8 \
    --model_dir=${MODEL} \
    --image_file=${IMG}
