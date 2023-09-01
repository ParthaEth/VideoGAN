#!/bin/bash

# Specify the root directory where you want to start searching
root_directory='/is/cluster/fast/scratch/ssanyal/video_gan/fashion_videos/GENERATED_VIDEOS_3'
dest_root="$root_directory"_kp
# Create the directory and its parents if they don't exist
mkdir -p $dest_root

# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

run_id="$1"
total_runs=10

# Calculate start_id and end_id
start_id=$((run_id * total_runs))
end_id=$((start_id + total_runs))


# Get a sorted list of directories
sorted_directories=($(find "$root_directory" -mindepth 1 -maxdepth 1 -type d | sort))

# Ensure the starting index is within bounds
if [ "$start_id" -ge "${#sorted_directories[@]}" ]; then
    echo "Starting index is out of bounds."
    exit 1
fi

# Iterate from start_id to end_id (exclusive) in the sorted list of directories
for ((i = start_id; i < end_id && i < ${#sorted_directories[@]}; i++)); do
    vid="${sorted_directories[i]}"

    cd /is/cluster/pghosh/repos/never_eidt_from_cluster_remote_edit_loc/AlphaPose || exit 1
    # Check if this name is a directory in the first place
    if [ -d "$vid" ]; then
        /home/pghosh/miniconda3/envs/alphapose/bin/python scripts/demo_inference.py \
        --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth \
        --indir=${vid} --outdir=$dest_root/$(basename "$vid") --format open
    fi
done
