#!/bin/bash
CC="gcc-7"
#/home/pghosh/miniconda3/envs/VideoGan80GB/bin/python prune_bad_video.py --pid $1
/home/pghosh/miniconda3/envs/VideoGan80GB/bin/python prune_corrupted_directories_of_images.py --pid $1