import sys
sys.path.append('../')
import os
import shutil
import torch
import PIL
import numpy as np
import random

import torchvision
import tqdm
from training.triplane import OSGDecoder
from training.volumetric_rendering.renderer import AxisAligndProjectionRenderer
from training.dataset import ImageFolderDataset


class VidFromImg:
    def __init__(self, img_path, resolution, max_x_zoom=4, num_resolutions=20):
        image = PIL.Image.open(img_path).convert('RGB')
        image = image.resize((resolution, resolution))
        self.resolution = resolution

        target_resolution = np.round(np.linspace(resolution, resolution / max_x_zoom, num_resolutions)).astype(int)
        target_resolution = np.repeat(target_resolution, np.ceil(resolution / num_resolutions))[0:resolution]

        self.vid_vol = []
        for cur_res in target_resolution:
            frame = ImageFolderDataset._centre_crop_resize(image, cur_res, cur_res, resolution, resolution)
            frame = np.array(frame).transpose(2, 0, 1)
            self.vid_vol.append(frame)

        self.vid_vol = np.stack(self.vid_vol).astype(np.float32)/127.5 - 1
        self.vid_vol = self.vid_vol.transpose(1, 2, 3, 0)

    def get_slice(self, cond):
        const_ax_depth = int(cond[3] * self.resolution)
        if np.argmax(cond[:3]) == 0:
            return self.vid_vol[:, const_ax_depth:const_ax_depth+1, :, :].transpose(1, 0, 2, 3)
        elif np.argmax(cond[:3]) == 1:
            return self.vid_vol[:, :, const_ax_depth:const_ax_depth+1, :].transpose(2, 0, 1, 3)
        elif np.argmax(cond[:3]) == 2:
            return self.vid_vol[:, :, :, const_ax_depth:const_ax_depth+1].transpose(3, 0, 1, 2)


device = 'cuda'
plane_h = plane_w = 128
rendering_res = 256
plane_c = 16
num_planes = 18
planes = torch.randn(1, num_planes, plane_c, plane_h, plane_w, dtype=torch.float32).to(device)*0.01
planes.requires_grad = True
renderer = AxisAligndProjectionRenderer(return_video=True, num_planes=num_planes).to(device)
rend_params = [param for param in renderer.parameters()]


img_file = '/is/cluster/fast/pghosh/datasets/ffhq/256X256/00001.png'
out_dir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting'
os.makedirs(out_dir, exist_ok=True)
shutil.copyfile(img_file, os.path.join(out_dir, 'original_img.png'))


vid_vol = VidFromImg(img_file, rendering_res)
decoder = OSGDecoder(plane_c, {'decoder_lr_mul': 1, 'decoder_output_dim': 32}).to(device)
dec_params = [param for param in decoder.parameters()]

allparams = [planes, ] + rend_params + dec_params
opt = torch.optim.Adam(allparams, lr=1e-3, betas=(0.5, 0.99))

train_itr = 5_000
criterion = torch.nn.MSELoss()
axis_dict = {'x': [1, 0, 0], 'y': [0, 1, 0], 't': [0, 0, 1]}
pbar = tqdm.tqdm(range(train_itr))
losses = []
for i in pbar:
    cond_batch = []
    gt_img_batch = []
    constant_axis = random.choice(['x', 'y', 't'])
    # constant_axis = 't'
    # cnst_coordinate = np.random.uniform(0, 1, 1)[0]
    cnst_coordinate = random.choice(np.linspace(0.001, 0.999, rendering_res))
    # cnst_coordinate = random.choice(np.linspace(0.001, 0.999, 40))
    # import ipdb; ipdb.set_trace()
    # cnst_coordinate = 0.5
    cond = np.array(axis_dict[constant_axis] + [cnst_coordinate, ]).astype(np.float32)
    # print(f'cond:{cond}')
    gt_img = torch.from_numpy(vid_vol.get_slice(cond)).to(device)
    cond = torch.from_numpy(cond).to(device)

    feature_samples, _ = renderer(planes, decoder, cond[None, ...], None, {'density_noise': 0, 'box_warp': 0.999,
                                                                        'neural_rendering_resolution': rendering_res})

    # Reshape into 'raw' image
    feature_image = feature_samples.permute(0, 2, 1).reshape(1, feature_samples.shape[-1], rendering_res,
                                                             rendering_res).contiguous()
    # depth_image = depth_samples.permute(0, 2, 1).reshape(b_size, 1, H, W)

    # Run superresolution to get final image
    rgb_image = feature_image[:, :3]
    # import ipdb; ipdb.set_trace()
    loss = criterion(rgb_image, gt_img)
    loss.backward()
    # import ipdb; ipdb.set_trace()
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
    pbar.set_description(f'loss: {np.mean(losses[-100:]):0.6f}')

    if i % 200 == 0:
        cond = torch.tensor([[0, 0, 1, 0.1], [0, 0, 1, 0.9], [1, 0, 0, 0.5]], dtype=torch.float32, device=device)
        feature_samples, _ = renderer(planes.repeat(cond.shape[0], 1, 1, 1, 1), decoder, cond, None,
                                   {'density_noise': 0, 'box_warp': 0.999,
                                    'neural_rendering_resolution': rendering_res})
        # import ipdb; ipdb.set_trace()
        feature_image = feature_samples.permute(0, 2, 1).reshape(cond.shape[0], feature_samples.shape[-1],
                                                                 rendering_res, rendering_res).contiguous()
        rgb_image_to_save = torch.cat((feature_image[:, :3], gt_img), dim=0)
        torchvision.utils.save_image((rgb_image_to_save + 1)/2, os.path.join(out_dir, f'{i:03d}.png'))

# print(rend_params)
