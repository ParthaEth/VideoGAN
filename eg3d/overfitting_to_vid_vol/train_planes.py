import copy
import sys
import os
import argparse
import random

# Append the parent directory to sys.path to allow relative imports
sys.path.append('../')
sys.path.append('../../')

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
import dnnlib
from torchmetrics.image import StructuralSimilarityIndexMeasure


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
    const_ax_depth = int(cond[3] * vid_vol.shape[1 + np.argmax(cond[:3])])
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
        # constant_axis = random.choice(['x', 'y', 't'])
        constant_axis = 't'  # TODO(Partha): change following line if you change this
        # import ipdb; ipdb.set_trace()
        cnst_coordinate = random.choice(np.linspace(0.001, 0.999,
                                                    vid_vol.shape[1+ np.argmax(axis_dict[constant_axis])]))
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
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-r", "--rank", type=int, help="rank of this process")
    argParser.add_argument("-dp", "--disable_progressbar", type=lambda x: str(x).lower() in ('true', '1'),
                           default='False', help="don't show progress")
    argParser.add_argument("-vr", "--video_root", type=str, required=True, help="don't show progress")
    args = argParser.parse_args()

    axis_dict = {'x': [1, 0, 0], 'y': [0, 1, 0], 't': [0, 0, 1]}
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = 'cuda'

    network_pkl = None
    # network_pkl = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/' \
    #               '00004-ffhq-fasion_video_bdmm-gpus8-batch128-gamma1/network-snapshot-001228.pkl'
    if network_pkl is not None:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # import ipdb; ipdb.set_trace()
    motion_features = 36 if network_pkl is None else G.motion_features
    appearance_feat = 32 if network_pkl is None else G.appearance_feat
    plane_h = plane_w = 32 if network_pkl is None else G.backbone.generator.img_resolution
    rendering_res = 64 if network_pkl is None else G.neural_rendering_resolution
    target_resolution = 256
    # target_resolution = 64
    plane_c = appearance_feat + motion_features
    b_size = 16
    load_saved = False
    ssim = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(device)
    out_dir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting'
    os.makedirs(out_dir, exist_ok=True)
    # video_root = '/is/cluster/fast/pghosh/datasets/fasion_video_bdmm/'
    args.video_root = args.video_root.rstrip('/')
    videos = sorted(os.listdir(args.video_root))
    # import ipdb; ipdb.set_trace()
    random.Random(seed).shuffle(videos)
    video_path = os.path.join(args.video_root, videos[args.rank])
    vid_vol = read_video_vol(video_path)  # c, h, w, t
    num_missing_frames = 8
    total_frames = vid_vol.shape[-1]
    frame_start = np.random.randint(0, total_frames - num_missing_frames, 1)[0]
    missing_frame_ids = np.arange(frame_start, frame_start + num_missing_frames)
    print(f'inpenting frame: {missing_frame_ids}')
    missing_frame = copy.deepcopy(vid_vol[:, :, :, missing_frame_ids])
    missing_frame = torch.from_numpy(missing_frame).to(device).permute(3, 0, 1, 2)
    missing_frame_cond = []
    for missing_frame_id in missing_frame_ids:
        missing_frame_cond.append(axis_dict['t'] + [missing_frame_id / target_resolution, ])
    missing_frame_cond = torch.from_numpy(np.array(missing_frame_cond).astype(np.float32)).to(device)
    vid_vol[:, :, :, missing_frame_ids] = 999

    feature_grid_type = '3d_voxels'
    if feature_grid_type.lower() == 'triplane':
        feature_grid = torch.clip((1 / 30) * torch.randn(1, 1, plane_c, plane_h, plane_w,
                                                         dtype=torch.float32), min=-3, max=3).to(device)
    elif feature_grid_type.lower() == '3d_voxels':
        vol_h = vol_w = vol_d = int(np.power(plane_h * plane_w, 1/3))
        feature_grid = torch.clip((1 / 30) * torch.randn(1, plane_c, vol_h, vol_w, vol_d,
                                                         dtype=torch.float32), min=-3, max=3).to(device)
    elif feature_grid_type.lower() == 'positional_embedding':
        feature_grid = torch.zeros(1, 1, plane_c, plane_h, plane_w, dtype=torch.float32).to(device)

    ws = torch.clip(torch.randn(b_size + num_missing_frames, 10, 512, dtype=torch.float32), min=-3, max=3).to(device)
    feature_grid.requires_grad = True
    ws.requires_grad = True

    if load_saved:
        pre_trained = torch.load(os.path.join(out_dir, 'rend_and_dec_good.pytorch'))

    if network_pkl is None:
        renderer = AxisAligndProjectionRenderer(return_video=True, motion_features=motion_features,
                                                appearance_features=appearance_feat).to(device)
    else:
        renderer = G.renderer
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
        superresolution = SuperresolutionHybrid4X(channels=appearance_feat, img_resolution=target_resolution,
                                                  sr_num_fp16_res=4, sr_antialias=True, channel_base=32768,
                                                  channel_max=512, fused_modconv_default='inference_only').to(device)
        sup_res_params = [param for param in superresolution.parameters()]
    else:
        sup_res_params = []

    mdl_params = rend_params + dec_params + sup_res_params
    if feature_grid_type.lower() == 'positional_embedding':
        opt = torch.optim.Adam([{'params': ws, 'lr': 1e-3},
                                {'params': mdl_params, 'lr': 1e-3}], lr=1e-3, betas=(0.0, 0.9))
    else:
        opt = torch.optim.Adam([{'params': feature_grid, 'lr': 1e-2},
                                {'params': ws, 'lr': 1e-3},
                                {'params': mdl_params, 'lr': 1e-3}], lr=1e-3, betas=(0.0, 0.9))
    # import ipdb;ipdb.set_trace()
    # opt = torch.optim.Adam([{'params': planes, 'lr': 1e-3},
    #                         {'params': mdl_params}], lr=1e-3, betas=(0.0, 0.9))
    # opt = torch.optim.SGD(allparams, lr=1e-3,)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.25, patience=300, verbose=True,
                                                           threshold=1e-3)

    train_itr = 700
    criterion = torch.nn.MSELoss()
    if args.disable_progressbar:
        pbar = range(train_itr)
    else:
        pbar = tqdm.tqdm(range(train_itr))

    losses_sr = []
    losses_lr = []
    trn_best_psnr_lr = 0
    scale_down = torchvision.transforms.Resize(size=rendering_res,
                                               interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                               antialias=True)

    feature_grid = feature_grid.expand(b_size + num_missing_frames, -1, -1, -1, -1)
    # import ipdb; ipdb.set_trace()
    inpaint_mse_sr = []
    inpaint_mse_lr = []
    inpaint_psnr_lr_bst = 0
    inpaint_ssim = 0
    for i in pbar:
        # print(dec_params[0][0, 0, 0])
        # import ipdb; ipdb.set_trace()
        gt_img, cond = get_batch(vid_vol, b_size, device=device)

        # Concatinating the missing frame for evaluating missing frame inpainting error
        gt_img = torch.cat((gt_img, missing_frame), dim=0)
        cond = torch.cat((cond, missing_frame_cond), dim=0)

        valid_frame_mask = gt_img[:b_size].mean((1, 2, 3), keepdim=True) < 990
        # if any(valid_frame_mask == 0):
        #     # import ipdb; ipdb.set_trace()
        #     print(cond[:b_size][valid_frame_mask.squeeze() == 0] * 256)
        # import ipdb; ipdb.set_trace()
        colors_coarse, _, features, _ = \
            renderer(feature_grid, decoder, cond, None, {'density_noise': 0, 'box_warp': 0.999,
                                                   'neural_rendering_resolution': rendering_res,
                                                   'feature_grid_type': feature_grid_type})

        # Reshape into 'raw' image
        rgb_img = colors_coarse.permute(0, 2, 1).reshape(b_size + num_missing_frames,
                                                         colors_coarse.shape[-1], rendering_res, rendering_res)
        feature_image = features.permute(0, 2, 1).reshape(b_size + num_missing_frames,
                                                          features.shape[-1], rendering_res, rendering_res)

        # Run superresolution to get final image
        if target_resolution > rendering_res:
            sr_image = superresolution(rgb_img, feature_image, ws, noise_mode='none')
            # import ipdb; ipdb.set_trace()
            loss_sr = criterion(sr_image[:b_size] * valid_frame_mask, gt_img[:b_size] * valid_frame_mask)
            loss_lr = criterion(rgb_img[:b_size] * valid_frame_mask, scale_down(gt_img[:b_size] * valid_frame_mask))
        else:
            loss_sr = loss_lr = criterion(rgb_img[:b_size] * valid_frame_mask,
                                          scale_down(gt_img[:b_size] * valid_frame_mask))

        loss = (loss_sr + loss_lr) / 2
        loss.backward()
        # import ipdb; ipdb.set_trace()
        opt.step()
        opt.zero_grad()
        losses_sr.append(loss_sr.item())
        losses_lr.append(loss_lr.item())
        psnr_lr = 10 * np.log10(4 / np.mean(losses_lr[-10:]))

        # import ipdb; ipdb.set_trace()
        # inpaint_mse_sr.append(criterion(sr_image[b_size:], gt_img[b_size:]).item())
        inpaint_mse_lr.append(criterion(rgb_img[b_size:], scale_down(gt_img[b_size:])).item())
        inpaint_psnr_lr = 10 * np.log10(4 / np.mean(inpaint_mse_lr[-10:]))

        # inpaint_psnr_sr = 10 * np.log10(4 / np.mean(inpaint_mse_sr[-10:]))

        if inpaint_psnr_lr_bst < inpaint_psnr_lr:
            inpaint_psnr_lr_bst = inpaint_psnr_lr
            inpaint_ssim = ssim(rgb_img[b_size:], scale_down(gt_img[b_size:]))

        if trn_best_psnr_lr < psnr_lr and not args.disable_progressbar:
            trn_best_psnr_lr = psnr_lr
            torch.save({'renderer': renderer.state_dict(), 'decoder': decoder.state_dict()},
                       os.path.join(out_dir, 'rend_and_dec_not_useful.pytorch'))

        if not args.disable_progressbar:
            pbar.set_description(f'loss_sr: {np.mean(losses_sr[-10:]):0.6f}, loss_lr: {np.mean(losses_lr[-10:]):0.6f}, '
                                 f'trn_PSNR(lr): {psnr_lr:0.2f}, trn_PSNR_best(lr):{trn_best_psnr_lr:0.2f}, '
                                 # f'planes.std: {planes.std().item():0.5f}, '
                                 # f'planes.mean: {planes.mean().item():0.5f}, '
                                 f'inpaint_psnr_lr: {inpaint_psnr_lr:0.2f}, '
                                 # f'inpaint_psnr_sr: {inpaint_psnr_sr:0.2f}, '
                                 f'inpaint_psnr_lr_bst: {inpaint_psnr_lr_bst:2f}, '
                                 f'inp_ssim = {inpaint_ssim}')
        scheduler.step(np.mean(losses_sr[-10:]))

        if i % 200 == 0 and not args.disable_progressbar:
            cond = torch.tensor([[0, 0, 1, 0.1], [0, 0, 1, 0.9], [1, 0, 0, 0.5]], dtype=torch.float32, device=device)
            rgb_img_test, _, _, _ = renderer(feature_grid[:len(cond)], decoder, cond, None,
                                             {'density_noise': 0, 'box_warp': 0.999,
                                         'neural_rendering_resolution': rendering_res,
                                         'feature_grid_type': feature_grid_type})
            feature_image = rgb_img_test.permute(0, 2, 1).reshape(cond.shape[0], rgb_img_test.shape[-1], rendering_res,
                                                                  rendering_res)
            rgb_image_to_save = torch.cat((feature_image[:, :3], scale_down(gt_img[:3]),
                                           rgb_img[b_size:], scale_down(gt_img[b_size:])),
                                           dim=0)
            torchvision.utils.save_image((rgb_image_to_save + 1)/2, os.path.join(out_dir, f'{i:03d}.png'))

    # print(rend_params)
    np.savez(os.path.join(out_dir, f'{os.path.basename(args.video_root)}_{videos[args.rank]}_ststs.npz'),
             inpaint_psnr=inpaint_psnr_lr_bst,
             inpaint_ssim=inpaint_ssim.item())
