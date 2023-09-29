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

# train
root = '/is/cluster/fast/pghosh/datasets/sky_timelapse/sky_train'
# root = '/is/cluster/fast/pghosh/datasets/sky_timelapse_256'
# dest_root = '/is/cluster/fast/pghosh/datasets/sky_timelapse/video_clips'
dest_root = '/is/cluster/fast/pghosh/datasets/sky_timelapse/train_5th_frame'

os.makedirs(dest_root, exist_ok=False)

# test
# root = '/is/cluster/fast/pghosh/datasets/sky_timelapse/sky_test'
# dest_root = '/is/cluster/fast/pghosh/datasets/sky_timelapse/video_clips_test'

# nframes = 128
# max_clips_per_vid = 1
# frame_offset = 'random'

nframes = 32
max_clips_per_vid = None  # dumps all possible clips
frame_offset = None  # offset is set to 0
subsample_factor = 5

transforms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(360),
                                             torchvision.transforms.Resize(256),
                                             torchvision.transforms.ToTensor()])

data_set = VideoFolder(root, nframes, transforms, max_clips_per_vid=max_clips_per_vid, frame_offset=frame_offset,
                       subsample_factor=5)
dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=10)

vid_count = 0
max_vid_cnt = np.inf  # dumps all videos in dataset
# max_vid_cnt = 2049  # dumps all videos in dataset
for vid_id, (vid_b, label) in enumerate(tqdm.tqdm(dataloader)):
    video_dest_path = os.path.join(dest_root, f'{vid_id:05d}.mp4')
    write_video(video_dest_path, vid_b[0].permute(1, 2, 3, 0) * 255)  # centre top crop
    vid_count += 1
    if vid_count >= max_vid_cnt:
        break
