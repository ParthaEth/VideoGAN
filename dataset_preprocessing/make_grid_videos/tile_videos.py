import os

import imageio
import numpy as np
import tqdm
import random
from shutil import copyfile

# source_root = '/is/cluster/fast/pghosh/datasets/ffhq/256X256_zoom_vid'
# source_root = '/home/pghosh/Downloads/videoGeneration/bw_bouncing_sq/all_vids'
# source_root = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/ten_motions/00086-ffhq-ffhq_X_10_good_motions_10_motions-gpus8-batch32-gamma1/talking_faces_vid'
# source_root = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00004-ffhq-fasion_video_bdmm-gpus8-batch128-gamma1/gen_vids/'
# source_root = '/home/pghosh/Downloads/generated_videos/talking_faces/real'
source_root = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/webvid_10M_flowers/stg_t/random_crops/00005-ffhq--gpus8-batch128-gamma1/videos_001966_trunk_0.7'
# source_root = '/is/cluster/fast/pghosh/datasets/bouncing_sq/ranad_init_vel/vids'
# outdir = '/is/cluster/fast/pghosh/datasets/ffhq'
# outdir = '/home/pghosh/Downloads/videoGeneration/bw_bouncing_sq/'
outdir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/webvid_10M_flowers/stg_t/random_crops/good_motion/'
# outdir = '/home/pghosh/Downloads/generated_videos/talking_faces/'
# outdir = '/is/cluster/fast/pghosh/datasets/bouncing_sq/ranad_init_vel'

tile_width = 24
tile_height = 10
vid_h = vid_w = vid_frames = 256
seed = 0
copy = True

st_idx = 0
# copy_videos_row_col = [(0, 5), (0, 8), (0, 23), (3, 22), (4, 23), (8,16), (8, 22), (7, 5)]
# copy_videos_row_col = [(7, 5), (8, 3), (1, 14)]
copy_videos_row_col = [(0, 0), (9, 1)]
copy_for_vid_grid = [(0, 0), (0, 12), (2, 8), (2, 10), (2, 11), (2, 15), (2, 21), (3, 23),
                     (3, 19,), (4, 2), (4, 17), (4, 23), (5, 9), (5, 11), (5, 15), (5, 19),
                     (6, 4), (6, 20), (6, 13), (6, 22), (7, 4), (7, 13), (7, 16), (7, 23),
                     (8, 2), (8, 5), (8, 10), (8, 11), (8, 13), (8, 15), (8, 20), (9, 2),
                     (9,3), (10, 1), (10, 2), (10, 5), (9, 18)]
# sst_idx = 241
# copy_videos_row_col = [(1,9), (1, 10), (5, 21), ]
# copy_for_vid_grid = [(0, 2), (0, 3), (0, 6), (0, 14), (0, 16), (0, 18), (1, 4),
#                      (1, 15), (2, 7), (2, 14), (2, 20), (2, 23), (3, 1), (3, 4), (3, 6),
#                      (3, 11), (3, 19), (4, 3), (4, 7), (4, 15), (5, 13), (6, 9), (6, 12),
#                      (6, 18), (6, 22), (7, 20), (7, 22), (9, 3), (9, 11), (9, 12), (10, 2),
#                      (10, 6), (10, 8), (10, 22)]


vid_in_strm = [None] * tile_width * tile_height

video_out = imageio.get_writer(f'{outdir}/seed{seed}_start_{st_idx}_{tile_width}_{tile_height}.mp4', mode='I', fps=60,
                               codec='libx264')
all_vids = [vid_file for vid_file in sorted(os.listdir(source_root)) if vid_file.endswith('.mp4')]
random.Random(seed).shuffle(all_vids)
all_vids = all_vids[st_idx : st_idx + tile_width*tile_height]

for frm_id in tqdm.tqdm(range(160)):
    frame = np.zeros((vid_h * tile_height, vid_w * tile_width, 3), dtype=np.uint8)
    vid_id = 0
    for row in range(tile_height):
        for column in range(tile_width):
            if copy:
                if (row, column) in copy_videos_row_col:
                    copyfile(os.path.join(source_root, all_vids[vid_id]), os.path.join(outdir, all_vids[vid_id]))

            else:
                if vid_in_strm[vid_id] is None:
                    filename = os.path.join(source_root, all_vids[vid_id])
                    vid_in_strm[vid_id] = imageio.get_reader(filename)

                image = vid_in_strm[vid_id].get_data(frm_id)
                image[:, 0:1] = 255
                image[:, -2:] = 255
                image[0:1, :] = 255
                image[-2:, :] = 255

                frame[row * vid_h:row * vid_h + vid_h, column * vid_w:column * vid_w + vid_w] = image
            vid_id += 1
    if copy:
        break

    video_out.append_data(frame)

video_out.close()
