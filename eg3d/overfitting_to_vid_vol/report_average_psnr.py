import os
import numpy as np


root_dir_of_npys = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/'
npys = os.listdir(root_dir_of_npys)

accum_psnr = 0
accum_ssim = 0
num_files = 0
for npy_file in npys:
    if npy_file.endswith('.npz'):
        dat_tis_file = np.load(os.path.join(root_dir_of_npys, npy_file))
        print(f'psnr:{dat_tis_file["inpaint_psnr"]}, SSIM: {dat_tis_file["inpaint_ssim"]}')
        accum_psnr += dat_tis_file["inpaint_psnr"]
        accum_ssim += dat_tis_file["inpaint_ssim"]
        num_files += 1

print(f'Average interpolation PSNR of missing frames = {accum_psnr / num_files}, average interpolation ssim = '
      f'{accum_ssim / num_files} using {num_files} videos')
