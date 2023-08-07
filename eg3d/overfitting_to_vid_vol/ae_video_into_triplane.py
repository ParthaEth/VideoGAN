import sys

import ipdb

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
from torchvision.transforms import functional as F
from training.superresolution import SuperresolutionHybrid4X
from training.overfitting_to_vid_vol.tri_plane_encoder import TriplaneEncoder
from training.dataset import VideoFolderDataset
from torch_utils import misc


torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

device = 'cuda'
input_vid_res = 3, 256, 256, 256
tri_plane_res = 16, 64, 64
num_planes = 3
dataset_dir = '/is/cluster/fast/pghosh/datasets/ffhq/256X256/'
b_size = 32

encoder = TriplaneEncoder(input_vid_res, tri_plane_res)
enc_params = [param for param in encoder.parameters()]
renderer = AxisAligndProjectionRenderer(return_video=True, num_planes=num_planes).to(device)
rend_params = [param for param in renderer.parameters()]
decoder = OSGDecoder(tri_plane_res[0], {'decoder_lr_mul': 1, 'decoder_output_dim': 32}).to(device)
dec_params = [param for param in decoder.parameters()]

mdl_params = rend_params + enc_params
opt = torch.optim.Adam(mdl_params, lr=1e-3, betas=(0.0, 0.9))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.25, patience=300, verbose=True,
                                                       threshold=1e-3)

train_itr = 5000
criterion = torch.nn.MSELoss()
pbar = tqdm.tqdm(range(train_itr))
losses_sr = []
losses_lr = []
best_psnr = 0

training_set = VideoFolderDataset(data=dataset_dir, return_video=True, cache_dir=None,
                                  fixed_time_frames=True, time_steps=-1, blur_sigma=0)

training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=0, num_replicas=1, seed=0)
training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                         batch_size=b_size, pin_memory=False, prefetch_factor=1,
                                                         worker_init_fn=training_set.worker_init_fn))

for i in pbar:
    _, peep_vid_real, cond = next(training_set_iterator)
    planes_batch = encoder(peep_vid_real)
    rgb_img, features = renderer(planes_batch, decoder, cond, None,
                                 {'density_noise': 0, 'box_warp': 0.999,
                                  'neural_rendering_resolution': input_vid_res[1]})

    # Reshape into 'raw' image
    rgb_img = rgb_img.permute(0, 2, 1).reshape(b_size, rgb_img.shape[-1], rendering_res, rendering_res)   # resume writing from here!!!
    feature_image = features.permute(0, 2, 1).reshape(b_size, features.shape[-1], rendering_res, rendering_res)

    loss_sr = loss_lr = criterion(rgb_img, gt_img)

    loss = (loss_sr + loss_lr) / 2
    loss.backward()
    # import ipdb; ipdb.set_trace()
    opt.step()
    opt.zero_grad()
    losses_sr.append(loss_sr.item())
    losses_lr.append(loss_lr.item())
    psnr_lr = 10 * np.log10(4 / np.mean(losses_lr[-10:]))
    if best_psnr < psnr_lr:
        best_psnr = psnr_lr
        torch.save({'renderer': renderer.state_dict(), 'decoder': decoder.state_dict()},
                   os.path.join(out_dir, 'rend_and_dec_not_useful.pytorch'))

    pbar.set_description(f'loss_sr: {np.mean(losses_sr[-10:]):0.6f}, loss_lr: {np.mean(losses_lr[-10:]):0.6f}, '
                         f'PSNR: {psnr_lr:0.2f}, PSNR_best:{best_psnr:0.2f}, '
                         f'planes.std: {planes_batch[0, 0, 0].std().item():0.5f}, '
                         f'planes.mean: {planes_batch[0, 0, 0].mean().item():0.5f}')
    scheduler.step(np.mean(losses_sr[-10:]))

    if i % 200 == 0:
        cond = torch.tensor([[0, 0, 1, 0.1], [0, 0, 1, 0.9], [1, 0, 0, 0.5]], dtype=torch.float32, device=device)
        rgb_img, _ = renderer(planes_batch[:cond.shape[0]], decoder, cond, None,
                              {'density_noise': 0, 'box_warp': 0.999,
                                      'neural_rendering_resolution': rendering_res})
        feature_image = rgb_img.permute(0, 2, 1).reshape(cond.shape[0], rgb_img.shape[-1], rendering_res, rendering_res)
        rgb_image_to_save = torch.cat((feature_image[:, :3], scale_down(gt_img[:3])), dim=0)
        torchvision.utils.save_image((rgb_image_to_save + 1)/2, os.path.join(out_dir, f'{i:03d}.png'))

# print(rend_params)
