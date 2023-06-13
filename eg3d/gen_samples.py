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
import imageio


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from torchvision.utils import flow_to_image, make_grid


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


def get_identity_flow(rend_res, dtype, device):
    cod_x = cod_y = torch.linspace(-1, 1, rend_res, dtype=dtype, device=device)
    grid_x, grid_y = torch.meshgrid(cod_x, cod_y, indexing='ij')
    coordinates = torch.stack((grid_x, grid_y), dim=0)
    return coordinates

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--axis', help='Axis perpendicular to which to take the frames', metavar='STR',
              type=click.Choice(['x', 'y', 't']), required=False, default='t')
@click.option('--img_type', help='Make video of raw images or sr_iamges', metavar='STR',
              type=click.Choice(['raw', 'sr_image']), required=False, default='sr_image')

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
    axis: str,
    img_type:str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    if img_type.lower() == 'raw':
        img_type = 'image_raw'
    elif img_type.lower() == 'sr_image':
        img_type = 'image'

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

    # Generate batch of images.
    b_size = 4
    identity_grid = get_identity_flow(G.img_resolution, device=device, dtype=torch.float32)
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        video_out = imageio.get_writer(f'{outdir}/seed{seed:04d}.mp4', mode='I', fps=30, codec='libx264')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim,).astype(np.float32)).to(device).repeat(b_size, 1)
        time_cod = torch.linspace(0, 1, G.img_resolution)

        batches = G.img_resolution // b_size
        # x0 = np.random.randint(0, 256 - 40)/256
        # y0 = np.random.randint(0, 256 - 40)/256
        # vel_p_frame = [0, 0]
        sup_res_factor = G.img_resolution / G.neural_rendering_resolution
        for b_id in range(batches):

            conditioning_params = torch.zeros((b_size, G.c_dim), dtype=z.dtype, device=device)
            if axis == 'x':
                conditioning_params[:, 0] = 1
            elif axis == 'y':
                conditioning_params[:, 1] = 1
            elif axis == 't':
                conditioning_params[:, 2] = 1
            else:
                raise ValueError(f'Undefined axis: {axis}. Valid options are x, y, or t')

            conditioning_params[:, 3] = time_cod[b_id*b_size:b_id*b_size + b_size]
            # conditioning_params[:, 4:] = \
            #     torch.tensor([x0, y0, vel_p_frame[0], vel_p_frame[1]], dtype=torch.float32)[None, ...]

            # import ipdb; ipdb.set_trace()

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            g_out = G.synthesis(ws, conditioning_params, noise_mode='const')
            # import ipdb; ipdb.set_trace()
            img_batch = g_out[img_type]
            if b_id == 0:
                local_warped = img_batch

            img_batch = (img_batch * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)

            flows_and_masks = g_out['flows_and_masks'].reshape(b_size, 5, G.neural_rendering_resolution,
                                                               G.neural_rendering_resolution)
            flows_and_masks = torch.nn.functional.interpolate(flows_and_masks, img_batch.shape[-1])
            # import ipdb; ipdb.set_trace()
            local_flow = flow_to_image(identity_grid[None] - flows_and_masks[:, 0:2])
            global_flow = flow_to_image(identity_grid[None] - flows_and_masks[:, 2:4])
            mask = (flows_and_masks[:, 4:5] * 127.5 + 127.5).expand(-1, 3, -1, -1).to(torch.uint8)
            # import ipdb; ipdb.set_trace()

            high_res_flow = (identity_grid[None] -
                             (identity_grid[None] - flows_and_masks[:, 0:2])/sup_res_factor).permute(0, 2, 3, 1)
            for i in range(b_size):
                local_warped = torch.nn.functional.grid_sample(
                    local_warped[0:1].permute(0, 1, 3, 2),
                    high_res_flow[i:i+1],
                    align_corners=True,)
                # import ipdb; ipdb.set_trace()
                local_warped_clmp = (local_warped * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
                # mask 0 implies only local flow used
                video_frame = make_grid([local_warped_clmp[0], img_batch[i], local_flow[i], global_flow[i], mask[i]], 5,
                                        pad_value=255)
                video_out.append_data(video_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8))

        video_out.close()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
