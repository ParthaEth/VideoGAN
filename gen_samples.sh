#!/bin/bash
export CC="gcc-7"
echo $(($1*70))
echo $(($1*70+1))
echo "/home/pghosh/miniconda3/envs/VideoGan80GB/bin/python gen_samples.py --outdir=/is/cluster/fast/pghosh/datasets/eg3d_generated --network '/is/cluster/fast/pghosh/pre_trained/ffhq512-128.pkl'  --seeds 0-6"