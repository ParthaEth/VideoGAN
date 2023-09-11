#!/bin/bash

# Check if an argument is provided for run_id
if [ $# -eq 0 ]; then
    echo "Please provide a value for run_id."
    exit 1
fi

run_id="$1"
vids_p_run=20
offset=2001

# Calculate start_id and end_id
start_id=$((run_id * vids_p_run + offset))
end_id=$((start_id + vids_p_run))

CC=gcc-7

#/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T/bin/python gen_samples.py \
#--outdir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ten_motions/\
#00086-ffhq-ffhq_X_10_good_motions_10_motions-gpus8-batch32-gamma1/talking_faces_vid" \
#--network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ten_motions/\
#00086-ffhq-ffhq_X_10_good_motions_10_motions-gpus8-batch32-gamma1/network-snapshot-003680.pkl" \
#--seeds $start_id-$end_id --img_type sr_image --trunc=0.7 --show_flow False

/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
--outdir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/\
00004-ffhq-video_clips-gpus8-batch128-gamma1/videos" \
--network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00004-ffhq-video_clips-gpus8-batch128-gamma1/\
network-snapshot-009420.pkl" \
--seeds $start_id-$end_id --img_type sr_image --trunc=0.7 --show_flow False
