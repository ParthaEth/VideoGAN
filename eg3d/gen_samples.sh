#!/bin/bash
export CC="gcc-7"
vids_per_process=70
seed_start=$(($1*vids_per_process))
seed_end=$(($1*70+vids_per_process))
/home/pghosh/miniconda3/envs/VideoGan80GB/bin/python gen_samples.py --outdir=/is/cluster/fast/pghosh/datasets/eg3d_generated --network '/is/cluster/fast/pghosh/pre_trained/ffhq512-128.pkl'  --seeds $seed_start-$seed_end