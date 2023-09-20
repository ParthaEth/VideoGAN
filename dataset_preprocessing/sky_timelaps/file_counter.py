import os

# Define the root directory
# root_dir = "/is/cluster/fast/pghosh/datasets/sky_timelapse/sky_train"
root_dir = "/is/cluster/fast/pghosh/datasets/sky_timelapse_256"

# List of image file extensions you want to count
image_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]  # Add more if needed

# Initialize a counter for the total number of image files
total_image_count = 0

# Walk through the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # Check if the file has a valid image extension
        if filename.lower().endswith(tuple(image_extensions)):
            total_image_count += 1

# Print the total count of image files
print("Total image files:", total_image_count)