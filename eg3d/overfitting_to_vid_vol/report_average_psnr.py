import os
import numpy as np


root_dir_of_npys = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/'
npys = os.listdir(root_dir_of_npys)

accum_psnr = 0
num_files = 0
for npy_file in npys:
    if npy_file.endswith('.npy'):
        accum_psnr += np.load(os.path.join(root_dir_of_npys, npy_file))
        num_files += 1

print(f'Average interpolation PSNR of missing frames = {accum_psnr / num_files}, using {num_files} videos')
