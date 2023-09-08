#!/bin/bash

# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

run_id="$1"

CC=gcc-7

/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python train_planes.py -r $run_id -dp True \
-vr '/is/cluster/fast/pghosh/datasets/ffhq_X_10_good_motions_10_motions/'

/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python train_planes.py -r $run_id -dp True \
-vr '/is/cluster/fast/pghosh/datasets/fasion_video_bdmm/'

/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python train_planes.py -r $run_id -dp True \
-vr '/is/cluster/fast/pghosh/datasets/sky_timelapse/video_clips'
