import sys
sys.path.append('../../eg3d')
import os
import tqdm
import numpy as np
import PIL
import imageio
from torchvision.transforms import functional as F
import argparse


def dump_a_vid(img_path, vid_path, resolution, max_x_zoom=2, num_resolutions=256):
    image = PIL.Image.open(img_path).convert('RGB')
    # image = image.resize((resolution, resolution))
    resolution = resolution

    target_resolution = np.round(np.linspace(resolution, resolution / max_x_zoom, num_resolutions)).astype(int)
    target_resolution = np.repeat(target_resolution, np.ceil(resolution / num_resolutions))[0:resolution]
    # import ipdb; ipdb.set_trace()

    video_out = imageio.get_writer(vid_path, mode='I', fps=60, codec='libx264')

    try:
        for cur_res in target_resolution:
            # frame = ImageFolderDataset._centre_crop_resize(image, cur_res, cur_res, resolution, resolution)
            frame = F.resize(F.center_crop(image, cur_res), resolution)

            frame = np.array(frame) #.transpose(2, 0, 1)
            video_out.append_data(frame)

    finally:
        video_out.close()
    
    
if __name__ == '__main__':
    total_processes = 8
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-r", "--rank", type=int, help="rank of this process")
    args = argParser.parse_args()
    rank = args.rank
    st_end_idx = np.linspace(0, 70_000, total_processes).astype(int)
    st_idx = st_end_idx[rank]
    end_idx = st_end_idx[rank + 1]

    base_dir = '/is/cluster/fast/pghosh/datasets/ffhq/'
    img_dir = os.path.join(base_dir, '256X256')
    vid_dir = os.path.join(base_dir, '256X256_zoom_vid')
    print('Going to LS dir')
    os.makedirs(vid_dir, exist_ok=True)
    # files = sorted(os.listdir(img_dir))
    pbar = tqdm.tqdm(range(st_idx, end_idx))
    for idx in pbar:
        img_file = f'{idx:05d}.png'
        vid_file = f'{img_file[:-4]}.mp4'
        dump_a_vid(os.path.join(img_dir, img_file), os.path.join(vid_dir, vid_file), resolution=256)
        pbar.set_description(f'{idx}')

        

