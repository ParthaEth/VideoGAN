# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Code adapted from
"Alias-Free Generative Adversarial Networks"."""

import os
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from MultiMachineMPICluster.mmmc import launcher
import argparse


def launch_training(c, desc, outdir, dry_run, rank):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------


def init_dataset_kwargs(data, return_video, cache_dir, fixed_time_frames, num_frames):
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.VideoFolderDataset', path=data, use_labels=True,
                                     max_size=None, xflip=False, return_video=return_video, cache_dir=cache_dir,
                                     fixed_time_frames=fixed_time_frames, num_frames=num_frames)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
    dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
    dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
    dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
    return dataset_kwargs, dataset_obj.name

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------
def main(local_rank, global_rank, num_gpus):
    """Train a GAN using the techniques described in the paper
    """
    return_video = True
    num_frames = 16
    backbone_lr_mult = 1
    decoder_lr_mult = 1
    renderer_lr_mult = 1
    data = '/is/cluster/fast/pghosh/datasets/celebV_HQ/256X256X256'
    d_set_cache_dir = '/dev/shm/'
    fixed_time_frames = True
    cond = True
    mirror = False
    batch = 32
    cbase = 32768
    cmax = 512
    map_depth = None
    freezed = 0
    mbstd_group = None
    gamma = 1
    glr = None
    cfg = 'ffhq'
    dlr = 0.002
    metrics = 'none', 'fid10k, fid50k'
    kimg = 25000
    tick = 4
    snap = 20
    seed = 0
    workers = 16
    discrim_type = 'DualPeepDicriminator' #DualDiscriminator, AxisAlignedDiscriminator, DualPeepDicriminator
    disc_c_noise = 0
    force_sr_module = None
    gen_pose_cond = False
    gpc_reg_prob = 0.5
    c_scale = None
    sr_noise_mode = None
    density_reg = 0.25
    density_reg_p_dist = 0.004
    reg_type = 'l1'
    density_reg_every = 4
    blur_fade_kimg = 200
    gpc_reg_fade_kimg = 1000
    neural_rendering_resolution_initial = 64
    neural_rendering_resolution_final = None
    neural_rendering_resolution_fade_kimg = 1000
    sr_num_fp16_res = 4
    style_mixing_prob = 0
    aug = 'noaug'
    target = 0.6
    aug_fix_prob = 0.2
    resume = None  # path to checkpoint to resume from
    resume_blur = False
    g_num_fp16_res = 0
    d_num_fp16_res = 4
    nobench = True
    desc_add = ''  #string to include in the result directory name
    dry_run = False
    outdir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/'


    # Initialize config.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, return_video=return_video,
                                 mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', num_vid_frames=num_frames,
                                 block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(),
                                 epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8,
                                     backbone_lr_mult=backbone_lr_mult, decoder_lr_mult=decoder_lr_mult,
                                     renderer_lr_mult=renderer_lr_mult)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=False, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(
        data=data, return_video=return_video, cache_dir=d_set_cache_dir,
        fixed_time_frames=fixed_time_frames, num_frames=num_frames)
    if cond and not c.training_set_kwargs.use_labels:
        raise ValueError('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = cond
    c.training_set_kwargs.xflip = mirror
    c.training_set_kwargs.return_video = return_video


    # Hyperparameters & settings.
    c.num_gpus = num_gpus
    c.batch_size = batch
    c.batch_gpu = batch // num_gpus
    c.G_kwargs.channel_base = cbase
    c.D_kwargs.channel_base = 32768
    c.G_kwargs.channel_max = cmax
    c.D_kwargs.channel_max = 512
    c.G_kwargs.mapping_kwargs.num_layers = map_depth
    c.D_kwargs.block_kwargs.freeze_layers = freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = mbstd_group
    c.loss_kwargs.r1_gamma = gamma
    c.G_opt_kwargs.lr = (0.002 if cfg == 'stylegan2' else 0.0025) if glr is None else glr
    c.D_opt_kwargs.lr = dlr
    c.metrics = parse_comma_separated_list(metrics)
    c.total_kimg = kimg
    c.kimg_per_tick = tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = snap
    c.random_seed = c.training_set_kwargs.random_seed = seed
    c.data_loader_kwargs.num_workers = workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise ValueError('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise ValueError('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = 'training.triplane.TriPlaneGenerator'
    c.D_kwargs.class_name = f'training.dual_discriminator.{discrim_type}'
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = disc_c_noise # Regularization for discriminator pose conditioning

    if c.training_set_kwargs.resolution == 512:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
    elif c.training_set_kwargs.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"

    if force_sr_module != None:
        sr_module = force_sr_module
    
    rendering_options = {
        'image_resolution': c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': gpc_reg_prob,
        'c_scale': c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': density_reg, # strength of density regularization
        'density_reg_p_dist': density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': reg_type, # for experimenting with variations on density regularization
        'sr_antialias': True,
        'time_steps': num_frames,
    }

    if cfg == 'ffhq':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 2.25, # near point along each ray to start taking samples.
            'ray_end': 3.3, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
        })
    elif cfg == 'afhq':
        rendering_options.update({
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    elif cfg == 'shapenet':
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            'ray_start': 0.1,
            'ray_end': 2.6,
            'box_warp': 1.6,
            'white_back': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
        })
    else:
        assert False, "Need to specify config"



    if density_reg > 0:
        c.G_reg_interval = density_reg_every
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.num_fp16_res = 0
    c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * blur_fade_kimg / 32 # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = gpc_reg_prob if gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = True
    c.loss_kwargs.neural_rendering_resolution_initial = neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
    c.G_kwargs.sr_num_fp16_res = sr_num_fp16_res

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=cbase, channel_max=cmax, fused_modconv_default='inference_only')

    c.loss_kwargs.style_mixing_prob = style_mixing_prob

    # Augmentation.
    if aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if aug == 'ada':
            c.ada_target = target
        if aug == 'fixed':
            c.augment_p = aug_fix_prob

    # Resume.
    if resume is not None:
        c.resume_pkl = resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        if not resume_blur:
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if d_num_fp16_res > 0 else None

    if nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if desc_add is not None:
        desc += f'-{desc_add}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=outdir, dry_run=dry_run, rank=global_rank)

#----------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpn', '--gpu_per_node', type=int, default=1)
    parser.add_argument('-n', '--nodes', type=int, default=1)
    parser.add_argument('-nr', '--node_rank', type=int, default=0)
    parser.add_argument('-od', '--output_dir', type=str, default=None)
    parser.add_argument('-uid', '--unique_id', type=str, default=None)
    args = parser.parse_args()

    train_loop_kwargs = {'num_gpus': args.gpu_per_node * args.nodes}
    launcher.manage_com_run_train_loop(args.node_rank, args.output_dir, args.unique_id, args.gpu_per_node,
                                       args.nodes, main, **train_loop_kwargs)

#----------------------------------------------------------------------------
