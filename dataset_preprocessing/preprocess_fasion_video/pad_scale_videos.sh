#!/bin/bash

input_directory="/is/cluster/fast/pghosh/datasets/FashionVideo/test_train_comb/"
output_directory="/is/cluster/fast/pghosh/datasets/FashionVideo/test_train_comb_256X256X256/"

mkdir -p "$output_directory"

for input_file in "$input_directory"/*.mp4; do
    output_file="$output_directory/$(basename "$input_file")"
    ffmpeg -i "$input_file" -vf "pad=max(iw\,ih):max(iw\,ih):(ow-iw)/2:(oh-ih)/2:white,select=lt(n\,256),scale=256:256:force_original_aspect_ratio=increase,select='lt(n,256)'" -c:a copy "$output_file"
done