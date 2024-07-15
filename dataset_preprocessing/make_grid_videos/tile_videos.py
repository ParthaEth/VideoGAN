import os

import imageio
import numpy as np
import tqdm
import random

# source_root = '/is/cluster/fast/pghosh/datasets/ffhq/256X256_zoom_vid'
# source_root = '/home/pghosh/Downloads/videoGeneration/bw_bouncing_sq/all_vids'
# source_root = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/ten_motions/00086-ffhq-ffhq_X_10_good_motions_10_motions-gpus8-batch32-gamma1/talking_faces_vid'
# source_root = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00004-ffhq-fasion_video_bdmm-gpus8-batch128-gamma1/gen_vids/'
source_root = '/home/pghosh/Downloads/generated_videos/talking_faces/real'
# source_root = '/is/cluster/fast/pghosh/datasets/bouncing_sq/ranad_init_vel/vids'
# outdir = '/is/cluster/fast/pghosh/datasets/ffhq'
# outdir = '/home/pghosh/Downloads/videoGeneration/bw_bouncing_sq/'
outdir = '/home/pghosh/Downloads/generated_videos/talking_faces/'
# outdir = '/is/cluster/fast/pghosh/datasets/bouncing_sq/ranad_init_vel'
st_idx = 0
tile_width = 6
tile_height = 5
vid_h = vid_w = vid_frames = 256
seed = 0
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
            if vid_in_strm[vid_id] is None:
                filename = os.path.join(source_root, all_vids[vid_id])
                vid_in_strm[vid_id] = imageio.get_reader(filename)

            image = vid_in_strm[vid_id].get_data(frm_id)
            image[:, 0:1] = 255
            image[:, -2:] = 255
            image[0:1, :] = 255
            image[-2:, :] = 255

            vid_id += 1

            frame[row * vid_h:row * vid_h + vid_h, column * vid_w:column * vid_w + vid_w] = image

    video_out.append_data(frame)

video_out.close()
