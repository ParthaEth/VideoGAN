#!/bin/bash
CC="gcc-7"

run_id=$1

/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T/bin/python train.py \
--outdir='/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse' --cfg=ffhq \
--data='/is/cluster/fast/pghosh/datasets/sky_timelapse/video_clips' --gamma=1 --return_video=True \
--snap=20 --metrics=fid10k --workers=12 --gen_pose_cond=False --discrim_type=DualPeepDicriminator \
--fixed_time_frames True  --gpus=8 --batch=128 --blur_sigma=10
--d_set_cache_dir=/dev/shm
