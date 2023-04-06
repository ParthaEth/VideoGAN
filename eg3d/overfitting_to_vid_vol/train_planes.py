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


class VidFromImg:
    def __init__(self, img_path, resolution, max_x_zoom=4, num_resolutions=20):
        image = PIL.Image.open(img_path).convert('RGB')
        image = image.resize((resolution, resolution))
        self.resolution = resolution

        target_resolution = np.round(np.linspace(resolution, resolution / max_x_zoom, num_resolutions)).astype(int)
        target_resolution = np.repeat(target_resolution, np.ceil(resolution / num_resolutions))[0:resolution]

        self.vid_vol = []
        for cur_res in target_resolution:
            # frame = VideoFolderDataset._centre_crop_resize(image, cur_res, cur_res, resolution, resolution)
            frame = F.resize(F.center_crop(image, cur_res), resolution)
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


def get_batch(vid_vols, device):
    cond_batch = []
    gt_img_batch = []
    for vid_vol in vid_vols:
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
        cond_batch.append(cond)
        gt_img_batch.append(gt_img)

    return torch.cat(gt_img_batch), torch.stack(cond_batch)


torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
device = 'cuda'
plane_h = plane_w = 128
rendering_res = 256
target_resolution = 256
plane_c = 32
num_planes = 6
b_size = 20
toral_batches = 1
load_saved = False
out_dir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting'
os.makedirs(out_dir, exist_ok=True)

planes = torch.clip((1/10)*torch.randn(b_size*toral_batches, num_planes, plane_c, plane_h, plane_w,
                                dtype=torch.float32), min=-3, max=3).to(device)
ws = torch.clip(torch.randn(b_size*toral_batches, 10, 512, dtype=torch.float32), min=-3, max=3).to(device)
# planes = torch.ones(b_size, num_planes, plane_c, plane_h, plane_w, dtype=torch.float32).to(device) * 0.168 * 3
planes.requires_grad = True
ws.requires_grad = True

if load_saved:
    pre_trained = torch.load(os.path.join(out_dir, 'rend_and_dec_good.pytorch'))

renderer = AxisAligndProjectionRenderer(return_video=True, num_planes=num_planes).to(device)
rend_params = [param for param in renderer.parameters()]
if load_saved:
    renderer.load_state_dict(pre_trained['renderer'])
    for param in renderer.parameters():
        param.requires_grad = False

img_dir = '/is/cluster/fast/pghosh/datasets/ffhq/256X256/'
vid_vols = []
img_offset = 200
for img_id in range(b_size * toral_batches):
    img_file = os.path.join(img_dir, f'{img_offset + img_id:05d}.png')

    # shutil.copyfile(img_file, os.path.join(out_dir, 'original_img.png'))
    vid_vols.append(VidFromImg(img_file, target_resolution))
vid_vols = np.array(vid_vols)

decoder = OSGDecoder(plane_c, {'decoder_lr_mul': 1, 'decoder_output_dim': 32}).to(device)
dec_params = [param for param in decoder.parameters()]
if load_saved:
    decoder.load_state_dict(pre_trained['decoder'])
    for param in decoder.parameters():
        param.requires_grad = False

if target_resolution > rendering_res:
    superresolution = SuperresolutionHybrid4X(channels=32, img_resolution=target_resolution, sr_num_fp16_res=4,
                                              sr_antialias=True, channel_base=32768, channel_max=512,
                                              fused_modconv_default='inference_only').to(device)
    sup_res_params = [param for param in superresolution.parameters()]
else:
    sup_res_params = []

mdl_params = rend_params + dec_params + sup_res_params
opt = torch.optim.Adam([{'params': planes, 'lr': 0.1},
                        {'params': ws, 'lr': 1e-3},
                        {'params': mdl_params}], lr=1e-3, betas=(0.0, 0.9))
# import ipdb;ipdb.set_trace()
# opt = torch.optim.Adam([{'params': planes, 'lr': 1e-3},
#                         {'params': mdl_params}], lr=1e-3, betas=(0.0, 0.9))
# opt = torch.optim.SGD(allparams, lr=1e-3,)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.25, patience=300, verbose=True,
                                                       threshold=1e-3)

train_itr = 5000
criterion = torch.nn.MSELoss()
axis_dict = {'x': [1, 0, 0], 'y': [0, 1, 0], 't': [0, 0, 1]}
pbar = tqdm.tqdm(range(train_itr))
losses_sr = []
losses_lr = []
best_psnr = 0
scale_down = torchvision.transforms.Resize(size=rendering_res,
                                           interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                           antialias=True)
for i in pbar:
    # print(dec_params[0][0, 0, 0])
    batch_idx = np.random.randint(0, toral_batches * b_size, b_size)
    # import ipdb; ipdb.set_trace()
    gt_img, cond = get_batch(vid_vols=vid_vols[batch_idx], device=device)
    planes_batch = planes[batch_idx].to(device)
    rgb_img, features = renderer(planes_batch, decoder, cond, None, {'density_noise': 0, 'box_warp': 0.999,
                                                                'neural_rendering_resolution': rendering_res})

    # Reshape into 'raw' image
    rgb_img = rgb_img.permute(0, 2, 1).reshape(b_size, rgb_img.shape[-1], rendering_res, rendering_res)
    feature_image = features.permute(0, 2, 1).reshape(b_size, features.shape[-1], rendering_res, rendering_res)

    # Run superresolution to get final image
    ws_batch = ws[batch_idx].to(device)
    if target_resolution > rendering_res:
        sr_image = superresolution(rgb_img, feature_image, ws_batch, noise_mode='none')
        # import ipdb; ipdb.set_trace()
        loss_sr = criterion(sr_image, gt_img)
        loss_lr = criterion(rgb_img, scale_down(gt_img))
    else:
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
