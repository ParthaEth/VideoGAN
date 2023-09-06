import torchvision
import os
import numpy as np
import imageio
import tqdm
from torch.utils.data import DataLoader
from video_folder import VideoFolder


def write_video(video_dest_path, vid):
    vid = vid.numpy().astype(np.uint8)
    video_out = imageio.get_writer(video_dest_path, mode='I', fps=30, codec='libx264')
    for video_frame in vid:
        video_out.append_data(video_frame)
    video_out.close()


root = '/is/cluster/fast/pghosh/datasets/sky_timelapse/sky_train'
dest_root = '/is/cluster/fast/pghosh/datasets/sky_timelapse/video_clips'
nframes = 32
training_data = VideoFolder(root, nframes, torchvision.transforms.ToTensor())
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=10)

for vid_id, (vid_b, label) in enumerate(tqdm.tqdm(train_dataloader)):
    video_dest_path = os.path.join(dest_root, f'{vid_id:05d}.mp4')
    write_video(video_dest_path, vid_b[0].permute(1, 2, 3, 0)[:, :256, 192:448] * 255)
