#!/bin/bash
source /home/pghosh/miniconda3/etc/profile.d/conda.sh
conda activate VideoGan80GB_STG_T_2
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
trunc_coeffs=("0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2")
#trunc_coeffs=("0.9" "0.8" "1.0")
#trunc_coeffs=("0.9")
for trunc in "${trunc_coeffs[@]}"; do

   echo "run_id = flowers_10M"
   out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/webvid_10M_flowers/random_crops/00005-ffhq--gpus8-batch128-gamma1/videos_001966_trunk_$trunc"
   network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/webvid_10M_flowers/random_crops/00005-ffhq--gpus8-batch128-gamma1/network-snapshot-001966.pkl"
   cfg="ffhq"

   /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
   --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
   --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True

#  echo "run_id = ffhqXcelebvHQ"
##  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/FFHQXcelebVHQ/00012-ffhq-ffhq_X_celebv_hq-gpus8-batch128-gamma1/videos_trunk_$trunc"
##  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/FFHQXcelebVHQ/00012-ffhq-ffhq_X_celebv_hq-gpus8-batch128-gamma1/network-snapshot-000983.pkl"
##  cfg="ffhq"
##
##  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
##  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
##  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True
#
#
#  echo "run_id = fashion_videos"
#  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00032-ffhq-fasion_video_bdmm_all-gpus4-batch256-gamma1/videos_000491_trunk_$trunc"
#  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00032-ffhq-fasion_video_bdmm_all-gpus4-batch256-gamma1/network-snapshot-000491.pkl"
#  cfg="ffhq"
#
#  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
#  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
#  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True
#
#  echo "run_id = fashion_videos"
#  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00032-ffhq-fasion_video_bdmm_all-gpus4-batch256-gamma1/videos_000573_trunk_$trunc"
#  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00032-ffhq-fasion_video_bdmm_all-gpus4-batch256-gamma1/network-snapshot-000573.pkl"
#  cfg="ffhq"
#
#  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
#  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
#  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True
#
#
#
##  echo "run_id = ucf"
##  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00026-ffhq-clips-gpus4-batch256-gamma1/videos_001638_trunk_$trunc"
##  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00026-ffhq-clips-gpus4-batch256-gamma1/network-snapshot-001638.pkl"
##  cfg="ffhq"
##
##  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
##  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
##  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True
##
#  echo "run_id = ffhq"
#  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/FFHQXcelebVHQ/00023-ffhq-ffhq_X_celebv_hq-gpus4-batch256-gamma1/videos_000491_trunk_$trunc"
#  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/FFHQXcelebVHQ/00023-ffhq-ffhq_X_celebv_hq-gpus4-batch256-gamma1/network-snapshot-000491.pkl"
#  cfg="ffhq"
#
#  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
#  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
#  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True
#
#  echo "run_id = ffhq"
#  out_dir="/is/cluster/fast/pghosh/ouputs/video_gan_runs/FFHQXcelebVHQ/00023-ffhq-ffhq_X_celebv_hq-gpus4-batch256-gamma1/videos_000573_trunk_$trunc"
#  network="/is/cluster/fast/pghosh/ouputs/video_gan_runs/FFHQXcelebVHQ/00023-ffhq-ffhq_X_celebv_hq-gpus4-batch256-gamma1/network-snapshot-000573.pkl"
#  cfg="ffhq"
#
#  /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python gen_samples.py \
#  --outdir=$out_dir --network=$network --seeds=$start_id-$end_id --img_type=sr_image --trunc=$trunc --show_flow=False \
#  --num_frames=$num_frames --reload_modules=False --cfg=$cfg --use_flow=True
done
