import torch
from typing import Dict
import os
import imageio
import numpy as np
import random
from scipy.ndimage import gaussian_filter


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                                           # Path to directory or zip.
        xflip=False,
        max_size=None,                                  # Artificially limit dataset size
        resolution=None,                                # Unused arg for backward compatibility
        load_n_consecutive: int=None,                   # Should we load first N frames for each video?
        load_n_consecutive_random_offset: bool=False,   # Should we use a random offset when loading consecutive frames?
        subsample_factor: int=1,                        # Sampling factor, i.e. decreasing the temporal resolution
        discard_short_videos: bool=False,               # Should we discard videos that are shorter than `load_n_consecutive`?
        blur_sigma: float=0,
        use_labels: bool=False,                         # will be silently ignored
        return_video: bool=True,
        cache_dir: str='None',
        fixed_time_frames: bool = True,
        time_steps: int = 0,
        apply_crop: bool = False,
        random_seed: int = 0,):                         # will be silently ignored
        super().__init__()
        self.blur_sigma = blur_sigma
        print(f'Video dataset with blur: {blur_sigma}, subsample_factor:{subsample_factor}')
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.resolution = resolution
        self.load_n_consecutive = load_n_consecutive
        self.load_n_consecutive_random_offset = load_n_consecutive_random_offset
        self.subsample_factor = subsample_factor
        assert not discard_short_videos, 'This flag has no effect. All videos must have 256 frames'
        assert max_size is None, 'This feature is not implemented, you specified max size of the dataset'
        assert not xflip, 'Doesn\'t amke any sense to flip video frames to compute FVD'

        if os.path.isdir(path):
            self._type = 'dir'
            self._all_fnames = [os.path.join(path, f_name)
                                for f_name in sorted(os.listdir(path)) if f_name.endswith('.mp4')]
            random.Random(3630).shuffle(self._all_fnames)

    def __getitem__(self, idx: int) -> Dict:
        fname = self._all_fnames[idx]
        reader = imageio.get_reader(fname, mode='I')

        self.start_frame_id = \
            np.random.randint(0, reader.count_frames() - self.load_n_consecutive * self.subsample_factor + 1)\
                if self.load_n_consecutive_random_offset else 0
        self.end_frame_id = self.load_n_consecutive * self.subsample_factor + self.start_frame_id

        vid_vol = []
        for fidx in range(self.start_frame_id, self.end_frame_id, self.subsample_factor):
            try:
                frame = reader.get_data(fidx)
                vid_vol.append(frame)
            except IndexError:
                raise IndexError(f'index {fidx} is out of total number of frames in the video '
                                 f'start_id{self.start_frame_id}, end id {self.end_frame_id}')
                break

        vid_vol = np.array(vid_vol)  # THWC
        # apply blur
        if self.blur_sigma >= 1e-2:
            vid_vol = gaussian_filter(vid_vol,
                                      sigma=(self.blur_sigma, self.blur_sigma),
                                      axes=(1, 2))

        vid_vol = vid_vol.transpose((0, 3, 1, 2))  # TCHW
        assert vid_vol.dtype == np.uint8

        return {
            'image': vid_vol,
            'label': [0, ] * len(vid_vol),
            'times': [0, ] * len(vid_vol),
            'video_len': len(vid_vol),
        }

    def __len__(self):
        return len(self._all_fnames)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_path = '/is/cluster/fast/pghosh/datasets/ffhqXcelebVhq_firstorder_motion_model/' \
                'ffhq_X_10_good_motions_10_motions/train'
    dataset_blr_5 = VideoFolderDataset(path=data_path, blur_sigma=5, load_n_consecutive=32)
    dataset_blr_0 = VideoFolderDataset(path=data_path, blur_sigma=0, load_n_consecutive=32)
    for i in range(500):
        img = dataset_blr_0.__getitem__(i)['image']
        img_blr = dataset_blr_5.__getitem__(i)['image']
        fig = plt.figure()
        ax1 = fig.add_subplot(121)  # left side
        ax2 = fig.add_subplot(122)  # right side
        ax1.imshow(img[20].transpose((1, 2, 0)))
        ax2.imshow(img_blr[20].transpose((1, 2, 0)))
        plt.show()
