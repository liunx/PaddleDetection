#!/bin/bash

MODEL=blazeface_1000e

python tools/export_model.py \
    -c configs/face_detection/${MODEL}.yml \
    -o weights=https://paddledet.bj.bcebos.com/models/${MODEL}.pdparams \
    --output_dir=inference_model
