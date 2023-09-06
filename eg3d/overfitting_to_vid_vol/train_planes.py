import sys
import os

# Append the parent directory to sys.path to allow relative imports
sys.path.append('../')

import shutil
import torch
import PIL
import numpy as np
import random

import imageio
import torchvision
import tqdm

from training.triplane import OSGDecoder
from training.volumetric_rendering.renderer import AxisAligndProjectionRenderer
from training.superresolution import SuperresolutionHybrid4X


def read_video_vol(vid_path):
    reader = imageio.get_reader(vid_path, mode='I')
    vid_vol = []
    for im in reader:
        vid_vol.append(im)

    vid_vol = np.array(vid_vol).astype(np.float32)/127.5 - 1  # t, h, w, c
    vid_vol = vid_vol.transpose((3, 1, 2, 0))  # c, h, w, t
    return vid_vol


def get_slice(vid_vol, cond):
    # vid_vol of shape  # c, h, w, t
    const_ax_depth = int(cond[3] * vid_vol.shape[1])
    if np.argmax(cond[:3]) == 0:
        return vid_vol[:, const_ax_depth:const_ax_depth+1, :, :].transpose(1, 0, 2, 3)  # 1, c, w, t
    elif np.argmax(cond[:3]) == 1:
        return vid_vol[:, :, const_ax_depth:const_ax_depth+1, :].transpose(2, 0, 1, 3)  # 1, c, h, t
    elif np.argmax(cond[:3]) == 2:
        return vid_vol[:, :, :, const_ax_depth:const_ax_depth+1].transpose(3, 0, 1, 2)  # 1, c, h, w


def get_batch(vid_vol, batch_size, device):
    cond_batch = []
    gt_img_batch = []
    for _ in range(batch_size):
        constant_axis = random.choice(['x', 'y', 't'])
        # constant_axis = 't'
        cnst_coordinate = random.choice(np.linspace(0.001, 0.999, rendering_res))
        # cnst_coordinate = random.choice(np.linspace(0.001, 0.999, 40))
        # import ipdb; ipdb.set_trace()
        # cnst_coordinate = 0.5
        cond = np.array(axis_dict[constant_axis] + [cnst_coordinate, ]).astype(np.float32)
        # print(f'cond:{cond}')
        gt_img = torch.from_numpy(get_slice(vid_vol, cond)).to(device)
        cond = torch.from_numpy(cond).to(device)
        cond_batch.append(cond)
        gt_img_batch.append(gt_img)

    return torch.cat(gt_img_batch), torch.stack(cond_batch)


if __name__ == '__main__':
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    device = 'cuda'
    motion_features = 9
    appearance_feat = 9
    plane_h = plane_w = 128
    rendering_res = 128
    target_resolution = 128
    plane_c = appearance_feat + motion_features
    b_size = 16
    toral_batches = 1
    load_saved = False
    out_dir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting'
    os.makedirs(out_dir, exist_ok=True)
    video_path = '/is/cluster/fast/pghosh/datasets/fasion_video_bdmm/91+20mY7UJS.mp4____91-2Jb8DkfS.mp4'
    vid_vol = read_video_vol(video_path)

    planes = torch.clip((1/10)*torch.randn(1, 1, plane_c, plane_h, plane_w,
                                           dtype=torch.float32), min=-3, max=3).to(device)
    ws = torch.clip(torch.randn(b_size*toral_batches, 10, 512, dtype=torch.float32), min=-3, max=3).to(device)
    planes.requires_grad = True
    ws.requires_grad = True

    if load_saved:
        pre_trained = torch.load(os.path.join(out_dir, 'rend_and_dec_good.pytorch'))

    renderer = AxisAligndProjectionRenderer(return_video=True, motion_features=motion_features).to(device)
    rend_params = [param for param in renderer.parameters()]
    if load_saved:
        renderer.load_state_dict(pre_trained['renderer'])
        for param in renderer.parameters():
            param.requires_grad = False

    decoder = OSGDecoder(appearance_feat, {'decoder_lr_mul': 1, 'decoder_output_dim': 32}).to(device)
    dec_params = [param for param in decoder.parameters()]
    if load_saved:
        decoder.load_state_dict(pre_trained['decoder'])
        for param in decoder.parameters():
            param.requires_grad = False

    if target_resolution > rendering_res:
        superresolution = SuperresolutionHybrid4X(channels=plane_c, img_resolution=target_resolution, sr_num_fp16_res=4,
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

    planes = planes.expand(b_size, -1, -1, -1, -1)
    # import ipdb; ipdb.set_trace()
    for i in pbar:
        # print(dec_params[0][0, 0, 0])
        batch_idx = np.random.randint(0, toral_batches * b_size, b_size)
        # import ipdb; ipdb.set_trace()
        gt_img, cond = get_batch(vid_vol, b_size, device=device)
        # planes_batch = planes[batch_idx].to(device)
        # import ipdb; ipdb.set_trace()
        colors_coarse, _, features, _ = \
            renderer(planes, decoder, cond, None, {'density_noise': 0, 'box_warp': 0.999,
                                                   'neural_rendering_resolution': rendering_res})

        # Reshape into 'raw' image
        rgb_img = colors_coarse.permute(0, 2, 1).reshape(b_size, colors_coarse.shape[-1], rendering_res, rendering_res)
        feature_image = features.permute(0, 2, 1).reshape(b_size, features.shape[-1], rendering_res, rendering_res)

        # Run superresolution to get final image
        ws_batch = ws[batch_idx].to(device)
        if target_resolution > rendering_res:
            sr_image = superresolution(rgb_img, feature_image, ws_batch, noise_mode='none')
            # import ipdb; ipdb.set_trace()
            loss_sr = criterion(sr_image, gt_img)
            loss_lr = criterion(rgb_img, scale_down(gt_img))
        else:
            loss_sr = loss_lr = criterion(rgb_img, scale_down(gt_img))

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
                             f'planes.std: {planes.std().item():0.5f}, '
                             f'planes.mean: {planes.mean().item():0.5f}')
        scheduler.step(np.mean(losses_sr[-10:]))

        if i % 200 == 0:
            cond = torch.tensor([[0, 0, 1, 0.1], [0, 0, 1, 0.9], [1, 0, 0, 0.5]], dtype=torch.float32, device=device)
            rgb_img, _, _, _ = renderer(planes[:len(cond)], decoder, cond, None,
                                        {'density_noise': 0, 'box_warp': 0.999,
                                         'neural_rendering_resolution': rendering_res})
            feature_image = rgb_img.permute(0, 2, 1).reshape(cond.shape[0], rgb_img.shape[-1], rendering_res,
                                                             rendering_res)
            rgb_image_to_save = torch.cat((feature_image[:, :3], scale_down(gt_img[:3])), dim=0)
            torchvision.utils.save_image((rgb_image_to_save + 1)/2, os.path.join(out_dir, f'{i:03d}.png'))

    # print(rend_params)
