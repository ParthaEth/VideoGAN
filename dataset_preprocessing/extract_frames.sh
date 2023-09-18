#!/bin/bash

# Set the source directory containing videos
source_dir="/is/cluster/fast/pghosh/datasets/ffhqXcelebVhq_firstorder_motion_model/ffhq_X_10_good_motions_10_motions/test"

# Set the destination directory for extracted frames
destination_dir="/is/cluster/fast/pghosh/datasets/ffhqXcelebVhq_firstorder_motion_model/ffhq_X_10_good_motions_10_motions/test_frames"

# Make sure the destination directory exists; create it if necessary
mkdir -p "$destination_dir"

# Function to calculate the time interval for frame extraction
calculate_time_interval() {
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1")
  interval=$(bc <<< "scale=2; $duration / 25")
  echo "$interval"
}

# Loop through each video file in the source directory
for video_file in "$source_dir"/*.mp4; do
  if [ -e "$video_file" ]; then
    # Extract frames from the video using FFmpeg
    filename=$(basename "$video_file")
    filename_noext="${filename%.*}"

    # Calculate the time interval for frame extraction
    interval=$(calculate_time_interval "$video_file")

    # Extract 25 frames equally spaced in time
    ffmpeg -i "$video_file" -vf "fps=1/$interval" "$destination_dir/${filename_noext}_frame_%04d.png"

    echo "Frames extracted from $filename and saved to $frame_dir"
  fi

done

echo "All frames extracted and saved to $destination_dir"
