#!/bin/bash
ARCH=arm
MODEL_DIR=$1

./opt_linux \
    --valid_targets=${ARCH} \
    --model_file=${MODEL_DIR}/model.pdmodel \
    --param_file=${MODEL_DIR}/model.pdiparams \
    --optimize_out=${MODEL_DIR}/model

python3 deploy/lite/convert_yml_to_json.py ${MODEL_DIR}/infer_cfg.yml