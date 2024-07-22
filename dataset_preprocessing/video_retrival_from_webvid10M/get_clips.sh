#!/bin/bash

# Check if the first argument is provided and is a numeric value
if [[ -z "$1" || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: Argument not provided or not a valid number."
    echo "Usage: $0 <numeric_argument>"
    exit 1
fi

# Run the Python script
/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python get_256X256_crops.py \
--run_id $1 \
--total_runs $2 \
--output_directory /is/cluster/fast/scratch/pghosh/dataset/webvid10M/flower_train_256X256X161_clips
