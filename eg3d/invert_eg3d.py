# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import imageio


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, RotateCamInCirclePerPtoZWhileLookingAt
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from training.perceptual_losses import perceptual_losses, loss_utils


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def get_camera_params(G, circle_radiuous, rot_angle, fov_deg, device):
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    # cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    cam2world_pose = RotateCamInCirclePerPtoZWhileLookingAt.sample(
        cam_pivot, z_dist=np.sqrt(cam_radius ** 2 - circle_radiuous ** 2), circle_radius=circle_radiuous,
        rot_angle=rot_angle, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot,
                                                           radius=cam_radius,
                                                           device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    return camera_params

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=False)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--img_type', help='Which image to optimize for sr_img/raw', type=str, required=False, default='raw', show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    img_type: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.sh.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)

    # TODO(Partha): Load image from disk
    if img_type.lower() == 'raw':
        image_type = 'image_raw'
        target_image = torch.randn((1, 3, 64, 64), device=device, dtype=torch.float32)
    elif img_type.lower() == 'sr_img':
        target_image = torch.randn((1, 3, 256, 256), device=device, dtype=torch.float32)
        image_type = 'image'

    # prepare perceptualossfunctions
    pretrained_perceptual_loss_weights = '/is/cluster/fast/pghosh/GIF_resources/input_files/perceptual_loss_resources' \
                                         '/resnet50_ft_weight.pkl'
    id_loss = perceptual_losses.VGGFace2Loss(pretrained_data=pretrained_perceptual_loss_weights)
    contextual_loss = perceptual_losses.VGGPerceptualLoss(resize=(target_image.shape[2] != 224),
                                                          vgg_type='vgg16',
                                                          perceptual=False,
                                                          contextual=True).to('cuda')
    l2 = torch.nn.MSELoss()
    loss_func = loss_utils.WeightedCombinationOfLosses(list_loss_funcs=[contextual_loss, id_loss, l2],
                                                       list_weights=[0.025, 0.025, 1])

    # Generate images.
    circle_radiuous = 0.5
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        out_dir_this_seed = os.path.join(outdir, f'seed{seed:04d}')
        # os.makedirs(out_dir_this_seed, exist_ok=True)
        vid_path = os.path.join(outdir, f'{seed:04d}_rgb.mp4')
        video_out = imageio.get_writer(vid_path, mode='I', fps=60, codec='libx264')

        # imgs = []
        angle_p = -0.2
        # yaw_angles = np.linspace(20/180*np.pi, -20/180*np.pi, 3)
        rot_angles = np.linspace(0, 2*np.pi, 240)[:-1]
        mse = torch.nn.MSELoss()
        optim_iter_num = 100

        camera_params = get_camera_params(G, circle_radiuous, 0, fov_deg, device)
        if G.c_dim > 25:
            print('Warning: Conditioning parms other than camera detected, shall we optimize for them?'
                  ' Now they are set to zeros')
            camera_params = torch.cat([camera_params,
                                             torch.zeros((1, G.c_dim - 25), device=device, dtype=torch.float32)],
                                            dim=1)

        ws = G.mapping(z, camera_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

        # optimize for given Image
        ws_opt = torch.tensor(data=ws.data, dtype=ws.dtype, device=ws.device, requires_grad=True)
        # TODO(Kamil): Add pivotal tuning parameters here
        optimizer = torch.optim.Adam(params=[ws_opt], lr=1e-3)
        pbar = tqdm(range(optim_iter_num))
        for optim_itr in pbar:
            synthesized = G.synthesis(ws_opt, camera_params[:25], noise_mode='const')
            img = synthesized[image_type]
            error = loss_func(img, target_image)

            pbar.set_description(
                f'optimizing, trn_loss: {error.item():0.3f}; mse:{mse(img, target_image).item():.3f}')

            error.backward()
            optimizer.step()
            optimizer.zero_grad()

        for i, rot_angle in enumerate(rot_angles):

            # img_depth = -synthesized['image_depth'][0]
            # img_depth = (img_depth - img_depth.min()) / (img_depth.max() - img_depth.min()) * 2 - 1
            camera_params = get_camera_params(G, circle_radiuous, rot_angle, fov_deg, device)
            synthesized = G.synthesis(ws_opt, camera_params[:25], noise_mode='const')
            img = synthesized['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(out_dir_this_seed,f'{i:04d}_rgb.png'))
            video_out.append_data(img[0].cpu().numpy())

            # img_depth = (img_depth.repeat(3, 1, 1).permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # PIL.Image.fromarray(img_depth.cpu().numpy(), 'RGB').save(os.path.join(out_dir_this_seed, f'{i:04d}_depth.png'))
        video_out.close()
        print(f'video saved at {vid_path}')
        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total=samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, camera_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
