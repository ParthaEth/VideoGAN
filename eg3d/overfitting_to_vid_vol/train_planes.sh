#!/bin/bash

# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

run_id="$1"

CC=gcc-7


datasets=\
("/is/cluster/fast/pghosh/datasets/ffhqXcelebVhq_firstorder_motion_model/ffhq_X_10_good_motions_10_motions/ffhq_X_10_good_motions_10_motions_all" \
"/is/cluster/fast/pghosh/datasets/fashion_videos/fasion_video_bdmm/fasion_video_bdmm_all" \
"/is/cluster/fast/pghosh/datasets/sky_timelapse/train_clips")

num_missing_frames=(3 8)

feature_grid_type=("triplane" "3d_voxels" "positional_embedding")

for dataset in "${datasets[@]}"; do
  for nfm in "${num_missing_frames[@]}"; do
    for fgt in "${feature_grid_type[@]}"; do
      /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python train_planes.py -r $run_id -dp True \
      -vr $dataset -nfm $nfm -fgt $fgt
    done
  done
done