import os
from PIL import Image
from argparse import ArgumentParser
import shutil
import numpy as np

parser = ArgumentParser()
parser.add_argument("--pid", type=int, help="path to config")
args = parser.parse_args()

src_dir = '/is/cluster/scratch/ssanyal/video_gan/fashion_videos/flower_train_vids_cnst_loc_watermark_256X256X32_crops/'
# src_dir = '/is/cluster/scratch/ssanyal/video_gan/fashion_videos/webvid10M_flowers'
n_files_per_process = 138
files_this_process = sorted(os.listdir(src_dir))
files_this_process = files_this_process[args.pid * n_files_per_process:args.pid * n_files_per_process + n_files_per_process]

for f_name in files_this_process:
    # print('Loop started')
    f_path = os.path.join(src_dir, f_name)
    if os.path.isdir(f_path):  # Check if it's a directory
        corrupted = False
        for root, dirs, files in os.walk(f_path):
            # print(f'checking {dirs}')
            if len(files) != 32:  # Check if there are 32 files in the directory
                corrupted = True
                break
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    # print(f'checking: {file_path}')
                    img = Image.open(file_path)
                    try:
                        img_np = np.array(img)
                        if len(img_np.shape) != 3 and img_np.shape[0] != 256 and\
                               img_np.shape[1] != 256 and img_np.shape[2] != 3:
                            corrupted = True
                            break
                    except:
                        corrupted = True
                        break
                    # Verify the integrity of the image file
                    # import ipdb; ipdb.set_trace()
                    if img.size != (256, 256) or img.mode != 'RGB':
                        corrupted = True
                        break
                except (OSError, IOError):
                    corrupted = True
                    break
            if corrupted:
                break
        if corrupted:
            shutil.rmtree(f_path)  # Remove the entire directory
            print(f'Removed directory: {f_path}')
        # else:
        #     print(f'No corrupted files in directory: {f_path}')
    else:  # If it's a file
        try:
            img = Image.open(f_path)
            # Verify the integrity of the image file
            img = img.transpose(2, 0, 1)
            if img.shape[1] != 3 or img.shape[2] != 256 or img.shape[3] != 256:
                os.remove(f_path)
        except (OSError, IOError):
            os.remove(f_path)
            print(f'Removed file: {f_path}')
