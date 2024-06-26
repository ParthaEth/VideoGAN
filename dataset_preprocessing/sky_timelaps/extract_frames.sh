#!/bin/bash

check_extracted_files() {
    local output_directory="$1"

    if [ ! -d "$output_directory" ]; then
        echo "Error: Output directory '$output_directory' does not exist."
        return
    fi

    for file in "$output_directory"/*.jpg; do
        if [ -f "$file" ] && [ ! -r "$file" ]; then
            echo "Error: File $file is not readable."
        fi
    done
}

# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

# Extract the run_id from the command-line argument
run_id="$1"

# Specify the root directory where your MP4 videos are located
#vid_src_dir="/is/cluster/fast/pghosh/datasets/sky_timelapse/video_clips"
#dest_root="/is/cluster/scratch/ssanyal/video_gan/fashion_videos/sky_timelapse"

#vid_src_dir="/is/cluster/fast/pghosh/datasets/sky_timelapse/train_clips"
#dest_root="/is/cluster/scratch/ssanyal/video_gan/fashion_videos/sky_timelapse_32frames"

#vid_src_dir="/is/cluster/fast/pghosh/datasets/fashion_videos/fasion_video_bdmm/fasion_video_bdmm_all"
#dest_root="/is/cluster/scratch/ssanyal/video_gan/fashion_videos/fashion_video_bdmm_all"

vid_src_dir="/is/cluster/fast/pghosh/datasets/ffhqXcelebVhq_firstorder_motion_model/ffhq_X_celebv_hq/"
dest_root="/is/cluster/scratch/ssanyal/video_gan/fashion_videos/ffhq_X_celebv_hq_all_new/"

if [ ! -d "$dest_root" ]; then
    mkdir "$dest_root"
fi

# Ensure the root directory exists
if [ ! -d "$vid_src_dir" ]; then
    echo "Root directory not found: $vid_src_dir"
    exit 1
fi

# Define the number of videos to process per run
vids_per_process=400   # Adjust as needed

# Calculate start_index and end_index based on run_id and vids_per_process
start_index=$((run_id * vids_per_process))
end_index=$((start_index + vids_per_process))

# Sort the MP4 files in the root directory and get them in an array
mapfile -t sorted_files < <(find "$vid_src_dir" -maxdepth 1 -type f -name "*.mp4" | sort)

# Iterate over the sorted files within the specified range
for ((i = start_index; i < end_index && i < ${#sorted_files[@]}; i++)); do
    video_file="${sorted_files[i]}"
    if [ -f "$video_file" ]; then
        # Extract the video file name (without extension) to create a directory
        video_name=$(basename "$video_file" .mp4)
        output_directory="$dest_root/$video_name"

        # Create a directory with the video name if it doesn't exist
        if [ ! -d "$output_directory" ]; then
            mkdir "$output_directory"
        fi

        # Use ffmpeg to extract frames into the corresponding directory
        ffmpeg -i "$video_file" -q:v 3 "$output_directory/frame%04d.jpg"
        chmod +w "$output_directory"/*.jpg
        check_extracted_files "$output_directory"

        # select every 5th frame
#        ffmpeg -i "$video_file" -q:v 3 -vf "select='not(mod(n\,5))'" -vframes 32 "$output_directory/frame%04d.jpg"


        echo "Extracted frames from $video_file to $output_directory/"
    fi
done

echo "Frame extraction complete."
