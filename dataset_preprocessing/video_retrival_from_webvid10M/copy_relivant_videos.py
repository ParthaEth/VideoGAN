import pandas as pd
import os
import shutil
import tqdm

# Paths to your CSV files and directories
video_ids_csv = '/is/cluster/scratch/pghosh/dataset/WebVid_10M/flowers_train_video_ids.csv'  # Path to your CSV with video IDs
main_csv = '/is/cluster/scratch/pghosh/dataset/WebVid_10M/results_10M_train.csv'  # Path to the main dataset CSV
source_directory = '/is/cluster/scratch/pghosh/dataset/WebVid_10M/train_videos/videos'  # Base directory containing video files
destination_directory = '/is/cluster/scratch/pghosh/dataset/WebVid_10M/flower_train_vids'  # Destination directory for copied files

# Read the relevant video IDs
relevant_ids = pd.read_csv(video_ids_csv, header=None, names=['videoid'])

# Read the main CSV data
main_data = pd.read_csv(main_csv)

# Merge to find corresponding page_dir for each relevant videoid
merged_data = pd.merge(relevant_ids, main_data, on='videoid', how='left')

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Copy the video files to the new directory
for _, row in tqdm.tqdm(merged_data.iterrows(), total=merged_data.shape[0]):
    source_file = os.path.join(source_directory, row['page_dir'], f"{row['videoid']}.mp4")
    destination_file = os.path.join(destination_directory, f"{row['videoid']}.mp4")
    # Check if source file exists before copying
    if os.path.exists(source_file):
        shutil.copy2(source_file, destination_file)
    #     print(f"Copied: {row['videoid']}.mp4")
    # else:
    #     print(f"File not found: {row['videoid']}.mp4")

print(f"All relevant files have been copied to {destination_directory}.")
