#!/bin/bash

# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

run_id="$1"

CC=gcc-7

/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python train_planes.py -r $run_id -dp True
