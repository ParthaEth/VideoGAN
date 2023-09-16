#!/bin/bash

# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

run_id="$1"
vids_p_run=21
offset=0

# Calculate start_id and end_id
start_id=$((run_id * vids_p_run + offset))
end_id=$((start_id + vids_p_run))

CC=gcc-7

#echo "run_id = 000"
#out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00000-ffhq-video_clips-gpus8-batch128-gamma1/\
#videos"
#network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00000-ffhq-video_clips-gpus8-batch128-gamma1/\
#network-snapshot-001228.pkl"
#
#/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
#--outdir=$out_dir --network=$network --seeds $start_id-$end_id --img_type sr_image --trunc=0.7 --show_flow False


echo "run_id = 007"
out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00007-ffhq-video_clips-gpus8-batch128-gamma1/\
videos_net_3440_fid_lowest_trunk0.9"
network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00007-ffhq-video_clips-gpus8-batch128-gamma1/\
network-snapshot-003440.pkl"

/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
--outdir=$out_dir --network=$network --seeds $start_id-$end_id --img_type sr_image --trunc=0.9 --show_flow False
