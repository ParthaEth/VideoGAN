# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random
import time
from pathlib import Path
import shutil
import imageio


try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache_dir = None,
        fixed_time_frames = False
    ):
        self.fixed_time_frames = fixed_time_frames
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        if cache_dir is None or cache_dir == 'None':
            self.cache_dir = None
        else:
            self.cache_dir = os.path.join(cache_dir, name)
            os.makedirs(self.cache_dir, exist_ok=True)

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def get_from_cached(self, fname):
        raise NotImplementedError

    def write_to_cache(self, image, fname):
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image, aug_label = self._load_raw_image(self._raw_idx[idx])
        # print(image.shape)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        label = self.get_label(idx)
        if len(label) > 0:
            label[0:4] = aug_label
            # if int(label[0]) != 2:
            #     print(f'label: {label}')
        return image.copy(), label

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        # if len(label) > 0 and int(label[0]) != 2:
        #     print(f'label: {label}')
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

    def worker_init_fn(self, w_id):
        seed = w_id + int(torch.randint(0, 10_000, (1,)))
        # print(f'dataloader seed set to {seed}')
        np.random.seed(seed)
        random.seed(seed)

#----------------------------------------------------------------------------

class VideoFolderDataset(Dataset):
    warning_displayed = 0
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        return_video    = False,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.return_video = return_video
        max_warnings = 1
        self.axis_dict = {'x': [1, 0, 0], 'y': [0, 1, 0], 't': [0, 0, 1]}

        if os.path.isdir(self._path):
            self._type = 'dir'
            # self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
            self._all_fnames = {os.path.join(self._path, f'{fname_idx:05d}.mp4') for fname_idx in range(50_000)}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        self._video_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in ['.mp4'])
        if len(self._video_fnames) == 0:
            raise IOError('No video files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._video_fnames)] + list(self._load_raw_image(0, skip_cache=True)[0].shape)
        # import ipdb; ipdb.set_trace()
        if raw_shape[1] != 3 and VideoFolderDataset.warning_displayed <= max_warnings:
            VideoFolderDataset.warning_displayed += 1
            print(f'\n\n\n\n\n\n\n\n\nWARNING{self.warning_displayed}: Generator cannot generate other than 3 color '
                  f'channel. But we got {raw_shape[1]} color cannels. forcing 3 color channels. This might be an '
                  f'error!!!')
            raw_shape[1] = 3
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def read_vid_from_file(self, fname):
        reader = imageio.get_reader(fname, mode='I')
        vid_vol = []
        for im in reader:
            vid_vol.append(im)

        return np.stack(vid_vol, axis=0).transpose(3, 1, 2, 0)

    def _load_raw_image(self, raw_idx, skip_cache=False):
        fname = self._video_fnames[raw_idx]
        lbl_cond = [0, 0, 1, 0]
        if getattr(self, 'cache_dir', None) is None or skip_cache:
            with self._open_file(fname) as f:
                vid_vol = self.read_vid_from_file(fname)
        else:
            vid_vol = self.get_from_cached(fname)
            if vid_vol is None:
                # print(f'Cache miss! {fname}')
                self.write_to_cache(fname)
                vid_vol = self.get_from_cached(fname)

        _, _, resolution, _ = vid_vol.shape
        if self.return_video:
            if getattr(self, 'fixed_time_frames', True):
                constant_axis = 't'
            else:
                constant_axis = random.choice(['x', 'y', 't'])
            cnst_coordinate = np.random.randint(0, resolution, 1)[0]
        else:
            constant_axis = 't'
            cnst_coordinate = 0

        # import ipdb; ipdb.set_trace()
        lbl_cond = self.axis_dict[constant_axis] + [cnst_coordinate/resolution]
        if constant_axis == 'x':
            image = vid_vol[:, cnst_coordinate, :, :]
        elif constant_axis == 'y':
            image = vid_vol[:, :, cnst_coordinate, :]
        elif constant_axis == 't':
            image = vid_vol[:, :, :, cnst_coordinate]

        return image, np.array(lbl_cond)

    def _load_raw_labels(self):
        # fname = 'dataset.json'
        # if fname not in self._all_fnames:
        #     return None
        # with self._open_file(fname) as f:
        #     labels = json.load(f)['labels']
        # if labels is None:
        #     return None
        # labels = dict(labels)
        # labels = [labels[fname.replace('\\', '/')] for fname in self._video_fnames]
        # labels = np.array(labels)
        # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])[:, :4]
        labels = np.zeros((len(self), 5), dtype=np.float32)
        if self.return_video:
            for i in range(len(labels)):
                if self.fixed_time_frames:
                    constant_axis = 't'
                else:
                    constant_axis = random.choice(['x', 'y', 't'])
                labels[i, :3] = np.array(self.axis_dict[constant_axis])
            cnst_coordinate = np.random.randint(0, self.resolution, len(labels))
            labels[:, 3] = cnst_coordinate / self.resolution
            # import ipdb; ipdb.set_trace()
        else:
            labels[:, 0:3] = np.array([[0, 0, 1], ]).repeat(len(labels), 0)
            labels[:, 3] = 0

        return labels

    def get_from_cached(self, fname):
        if self.cache_dir is None:
            return None
        else:
            file_path = os.path.join(self.cache_dir, fname)
            if os.path.exists(file_path):
                # cache hit
                try:
                    vid_vol = self.read_vid_from_file(file_path)
                except OSError as e:
                    print(f'Bad file {fname}. Removing from cache. \n {e}')
                    os.remove(file_path)
                    return None
                return vid_vol

            else:
                return None

    def write_to_cache(self, fname):
        if self.cache_dir is not None:
            dest_file_path = os.path.join(self.cache_dir, fname)
            if self.cache_dir is not None:
                try:
                    Path(dest_file_path).touch(exist_ok=False)
                    src_path = os.path.join(self._path, fname)
                    shutil.copyfile(src_path, dest_file_path)
                except FileExistsError:
                    pass
#----------------------------------------------------------------------------
