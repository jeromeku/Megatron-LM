#!/bin/bash

export CUDNN_ROOT=.venv/lib/python3.12/site-packages/nvidia/cudnn
export CPATH=$CUDNN_ROOT/include:$CPATH
export LIBRARY_PATH=$CUDNN_ROOT/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_ROOT/lib:$LD_LIBRARY_PATH


uv pip install transformer_engine[pytorch] --reinstall --torch-backend=cu128 --no-build-isolation 2>&1 | tee install.log