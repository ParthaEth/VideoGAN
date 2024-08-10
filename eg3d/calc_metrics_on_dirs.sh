#!/bin/bash
source /home/pghosh/miniconda3/etc/profile.d/conda.sh
conda activate VideoGan80GB_STG_T_2

data_2_dirs=('/is/cluster/scratch/ssanyal/video_gan/stylegan_V/web10mflowers/output/network-snapshot-02500_img')

for data_2_dir in "${data_2_dirs[@]}"; do
    /home/pghosh/miniconda3/envs/VideoGan80GB_STG_T_2/bin/python calc_metrics.py \
    --data=/is/cluster/fast/scratch/pghosh/dataset/webvid10M/flower_train_256X256X161_clips/ \
    --metrics=fvd2048_16f,fvd2048_128f,fid50k --gpus=4 --blur_sigma=0 --ssfd2=1 --cfg=ffhq \
    --data_2=$data_2_dir

    echo "Done with $data_2_dir"$'\n'$'\n'$'\n'
done
