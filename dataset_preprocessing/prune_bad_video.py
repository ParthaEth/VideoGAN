import os
import imageio
# import tqdm
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--pid", type=int, help="path to config")
args = parser.parse_args()

# src_dir = '/is/cluster/fast/pghosh/datasets/eg3d_generated_0.5'
# src_dir = '/is/cluster/fast/pghosh/datasets/ffhq_X_few_good_talking_motion_3_motions'
src_dir = '/is/cluster/fast/pghosh/datasets/fashion_ivd_first_order_motion'
# src_dir = '/dev/shm/eg3d_generated_0.5'
n_files_per_proces = 100
files_this_process = sorted(os.listdir(src_dir))
files_this_process = files_this_process[args.pid*n_files_per_proces:args.pid*n_files_per_proces + n_files_per_proces]

for f_name in files_this_process:
    f_path = os.path.join(src_dir, f_name)
    try:
        reader = imageio.get_reader(f_path, mode='I')
        if reader.get_data(238).shape == (256, 256, 3):
            continue
    except (OSError, IndexError):
        os.remove(f_path)
        print(f'removed {f_path}')