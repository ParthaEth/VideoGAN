#!/bin/bash
# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

run_id="$1"
sleep "$((run_id / 10))"

offset=3
run_id=$((run_id + offset))
# Specify the directory containing the .pkl files
directory="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00004-ffhq--gpus8-batch128-gamma1/"

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Directory does not exist: $directory"
  exit 1
fi

# List all .pkl files in the directory and sort them
pkl_files=($(find "$directory" -type f -name "*.pkl" | sort))


# Check if the index is within the bounds of the array
if [ "$run_id" -ge 0 ] && [ "$run_id" -lt "${#pkl_files[@]}" ]; then
  echo $run_id
  selected_pkl="${pkl_files[$run_id]}"
  echo "Processing ${selected_pkl}"
  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python calc_metrics.py \
  --data=/is/cluster/fast/pghosh/datasets/ucf101/clips/ \
  --network=$selected_pkl --metrics=fvd2048_16f,fvd2048_128f --gpus=4 --blur_sigma=0 --ssfd2=1 --cfg=ffhq --trunc=0.9
else
  echo "Index out of bounds: $run_id"
fi