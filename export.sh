#!/bin/bash

MODEL=picodet_s_320_voc

python tools/export_model.py \
    -c configs/picodet/${MODEL}.yml \
    -o weights=https://paddledet.bj.bcebos.com/models/${MODEL}.pdparams \
    --output_dir=inference_model
