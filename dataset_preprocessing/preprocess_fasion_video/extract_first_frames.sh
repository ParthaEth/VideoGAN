#!/bin/bash

input_directory="/is/cluster/fast/pghosh/datasets/FashionVideo/test_train_comb_256X256X256"
output_directory="/is/cluster/fast/pghosh/datasets/FashionVideo/test_train_comb_256X256X256_first_frames/"

mkdir -p "$output_directory"

for video_file in "$input_directory"/*.mp4; do
    video_filename=$(basename "$video_file")
    video_name="${video_filename%.*}"
    output_image="$output_directory/${video_name}.png"

    ffmpeg -i "$video_file" -vframes 1 -q:v 1 "$output_image"
done