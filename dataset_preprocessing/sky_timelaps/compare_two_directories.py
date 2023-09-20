import os

# Paths to the two parent directories
directory1 = "/is/cluster/fast/pghosh/datasets/sky_timelapse/sky_train"
directory2 = "/is/cluster/fast/pghosh/datasets/sky_timelapse_256"

# Get the list of directory names in the first directory
dirs1 = os.listdir(directory1)

# Get the list of directory names in the second directory
dirs2 = os.listdir(directory2)

# Find the directories that are in directory1 but not in directory2
difference1 = set(dirs1) - set(dirs2)

# Find the directories that are in directory2 but not in directory1
difference2 = set(dirs2) - set(dirs1)

if difference1:
    print("Directories in", directory1, "but not in", directory2, ":", difference1)

if difference2:
    print("Directories in", directory2, "but not in", directory1, ":", difference2)

if not difference1 and not difference2:
    print("No differing directory names found.")