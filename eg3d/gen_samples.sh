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

num_frames=160  # All others, then use sf ssfd2=5, in FVD computation
#num_frames=32  # Sky time-lapse only, in FVD computation
#trunc_coeffs=("0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2")
trunc_coeffs=("0.9" "0.8" "1.0")
for trunc in "${trunc_coeffs[@]}"; do
  #echo "run_id = 000"
  #out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00000-ffhq-video_clips-gpus8-batch128-gamma1/\
  #videos"
  #network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00000-ffhq-video_clips-gpus8-batch128-gamma1/\
  #network-snapshot-001228.pkl"
  #
  #/home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
  #--outdir=$out_dir --network=$network --seeds $start_id-$end_id --img_type sr_image --trunc=0.7 --show_flow False


  echo "run_id = ucf"
  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00012-ffhq-clips-gpus4-batch128-gamma1/1392/videos_trunk_$trunc"
  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00012-ffhq-clips-gpus4-batch128-gamma1/network-snapshot-001392.pkl"
  cfg="ffhq"

  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True

  echo "run_id = ucf"
  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00012-ffhq-clips-gpus4-batch128-gamma1/1556/videos_trunk_$trunc"
  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00012-ffhq-clips-gpus4-batch128-gamma1/network-snapshot-001556.pkl"
  cfg="ffhq"

  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True
done