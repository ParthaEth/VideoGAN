import torchvision
import os
import numpy as np
import imageio
import tqdm
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
from multiprocessing import Pool


def write_video(path_and_vid):
    video_dest_path, vid = path_and_vid
    vid = vid.numpy().astype(np.uint8)
    video_out = imageio.get_writer(video_dest_path, mode='I', fps=30, codec='libx264')
    for video_frame in vid:
        video_out.append_data(video_frame)
    video_out.close()


if __name__ == '__main__':
    saving_workers = 16
    time_steps = 160
    dataset_type = 'train'
    # dataset_type = 'test'
    # base_path = '/dev/shm/ucf/ucf101/'
    base_path = '/is/cluster/fast/pghosh/datasets/ucf101/'
    base_path_dest = '/is/cluster/fast/pghosh/datasets/ucf101/'
    root = os.path.join(base_path, 'UCF-101')
    annotation_path = os.path.join(base_path, 'ucfTrainTestlist')
    tforms = transforms.Compose([transforms.CenterCrop(240),
                                 transforms.Resize(256, antialias=True)])
    ucf_dataset = torchvision.datasets.UCF101(root, annotation_path, frames_per_clip=time_steps,
                                              step_between_clips=5, frame_rate=30, train=(dataset_type == 'train'),
                                              output_format='TCHW', transform=tforms, num_workers=8,)
    # dataloader = DataLoader(ucf_dataset, batch_size=saving_workers, shuffle=False, num_workers=8, prefetch_factor=2)

    vid_count = 0
    max_vid_cnt = np.inf  # dumps all videos in dataset
    # max_vid_cnt = 2049  # dumps all videos in dataset
    # import ipdb; ipdb.set_trace()
    clip_num = 0
    with Pool(saving_workers) as p:
        for batch_id in tqdm.tqdm(range(len(ucf_dataset) // saving_workers)):
            process_args = []
            for i in range(saving_workers):
                clip_num = batch_id * saving_workers + i
                vid, _, _ = ucf_dataset[clip_num]
                video_dest_path = os.path.join(base_path_dest, f'clips/{dataset_type}_{clip_num:05d}.mp4')
                video = copy.deepcopy(vid.permute(0, 2, 3, 1))
                process_args.append((video_dest_path, video))
            p.map(write_video, process_args)
            vid_count += 1
            if vid_count >= max_vid_cnt:
                break
