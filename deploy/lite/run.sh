#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./Libs

./build/main det_runtime_config.json $@
