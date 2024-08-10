import os
import random
import shutil
import argparse

def copy_random_files(source_dir, destination_dir, max_count=2048):
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # List all the subdirectories in the source directory
    count = 0
    for subdir in next(os.walk(source_dir))[1]:
        full_subdir_path = os.path.join(source_dir, subdir)

        # Get all image files in the subdirectory
        files = [f for f in os.listdir(full_subdir_path)
                 if os.path.isfile(os.path.join(full_subdir_path, f)) and
                 f.lower().endswith(('.jpg', '.png'))]

        if files:
            # Choose a random file
            random_file = random.choice(files)
            full_file_path = os.path.join(full_subdir_path, random_file)

            # Create a new filename based on the parent directory name
            new_filename = f"{subdir}_{random_file}"
            destination_file_path = os.path.join(destination_dir, new_filename)

            # Copy the file to the destination directory
            shutil.copy(full_file_path, destination_file_path)
            print(f"Copied {random_file} to {destination_file_path}")
            count += 1
        else:
            print(f"No image files found in {full_subdir_path}")

        if count >= max_count:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy random images from each subdirectory to a destination directory")
    parser.add_argument("source_dir", type=str, help="Path to the source directory")
    parser.add_argument("destination_dir", type=str, help="Path to the destination directory")
    args = parser.parse_args()

    # Call the function with command-line arguments
    copy_random_files(args.source_dir, args.destination_dir)
