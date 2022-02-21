#!/bin/bash

MODEL=picodet_mobilenetv3_large_1x_416_coco

python tools/export_model.py \
    -c configs/picodet/more_config/${MODEL}.yml \
    -o weights=https://paddledet.bj.bcebos.com/models/${MODEL}.pdparams \
    --output_dir=inference_model
