#!/bin/bash

# Define the root directory where you want to start the search
root_directory="./"

# Use the find command to locate all 'logs' directories
# and execute a command within each of them to delete their contents
find "$root_directory" -type d -name "logs" -exec sh -c 'rm -rf "$1"/*' shell {} \;

