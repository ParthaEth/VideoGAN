import os

import imageio
import numpy as np
import tqdm

source_root = '/is/cluster/fast/pghosh/datasets/ffhq/256X256_zoom_vid'
outdir = '/is/cluster/fast/pghosh/datasets/ffhq'
st_idx = 0
tile_width = tile_height = 5
vid_h = vid_w = vid_frames = 256
vid_in_strm = [None] * tile_width * tile_height

video_out = imageio.get_writer(f'{outdir}/seed{st_idx}_{tile_width}_{tile_height}.mp4', mode='I', fps=60,
                               codec='libx264')
for frm_id in tqdm.tqdm(range(256)):
    frame = np.zeros((vid_h * tile_height, vid_w * tile_width, 3), dtype=np.uint8)
    vid_id = 0
    for row in range(tile_height):
        for column in range(tile_width):
            if vid_in_strm[vid_id] is None:
                filename = os.path.join(source_root, f'{st_idx + vid_id:05d}.mp4')
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
