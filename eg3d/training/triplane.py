# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torchvision.transforms

from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
# from training.networks_stylegan_xl import UnifiedGenerator as StyleGANXLBackbone
from training.stylegan_t.generator import WrapperStyleGANXLLike as StyleGANTBackbone
from training.volumetric_rendering.renderer import AxisAligndProjectionRenderer
# from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        return_video = False,
        path_stem = None,           # Progressively growing from previous stem. this is path to that pretrained stem
        head_layers = None,         # How many head layers to add on top of the stem
        up_factor = None,           # What is the resolution up factor at this super res scale
        data_blur_sigma = None,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        # self.renderer = ImportanceRenderer()
        self.appearance_features = 32  # 32 appearance features 6 motion features
        self.motion_features = 36  # must be multiple of 3
        assert self.motion_features % 3 == 0
        self.num_planes = 1
        self.renderer = AxisAligndProjectionRenderer(return_video, self.motion_features)
        # self.renderer = ImportanceRenderer(self.neural_rendering_resolution, return_video)
        # self.ray_sampler = RaySampler()
        self.neural_rendering_resolution = 64
        # self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256,
        #                                   img_channels=self.appearance_features + self.motion_features,
        #                                   mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        if not isinstance(data_blur_sigma, str):
            data_blur_sigma = f'{data_blur_sigma:.2f}'
        # blur_to_res = {'10.00': 16, '5.00': 32, '2.50': 64, '1.25': 128, '0.00': 256}
        blur_to_res = {'10.00': 16, '5.00': 16, '2.50': 32, '1.25': 64, '0.00': 128}
        # self.backbone = StyleGANXLBackbone(z_dim, c_dim, w_dim, img_resolution=blur_to_res[data_blur_sigma],
        #                                    img_channels=self.appearance_features + self.motion_features,
        #                                    mapping_kwargs=mapping_kwargs, path_stem=path_stem, head_layers=head_layers,
        #                                    up_factor=up_factor, **synthesis_kwargs)
        self.backbone = StyleGANTBackbone(z_dim, c_dim=c_dim, w_dim=w_dim, img_resolution=blur_to_res[data_blur_sigma],
                                          img_channels=self.appearance_features + self.motion_features,
                                          conditional=False, path_stem=path_stem, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(self.appearance_features,
                                  {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1),
                                   'decoder_output_dim': 32})

        ########################### Load pre-trained ###################################################
        # pre_trained = torch.load('/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/'
        #                          'rend_and_dec_256_rend.pytorch')
        # self.renderer.load_state_dict(pre_trained['renderer'])
        # # for param in self.renderer.parameters():
        # #     param.requires_grad = False
        #
        # self.decoder.load_state_dict(pre_trained['decoder'])
        # # for param in self.renderer.parameters():
        # #     param.requires_grad = False
        ########################### Load pre-trained ################################################### 

        self.rendering_kwargs = rendering_kwargs
        self._last_planes = None
        self.downscale4x = torchvision.transforms.Resize(64, antialias=True)
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                cond = torch.zeros_like(c)
        else:
            cond = c.clone()
            # cond[:, :4] *= 0
        # import ipdb; ipdb.set_trace()
        return self.backbone.mapping(z, cond * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi,
                                     truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                  use_cached_backbone=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)
        # import ipdb; ipdb.set_trace()
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        self.rendering_kwargs['neural_rendering_resolution'] = neural_rendering_resolution
        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # torch.save(planes, '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/eg3d_init_planes.pth')
        # import ipdb; ipdb.set_trace()

        # Reshape output into three 32-channel planes
        b_size = len(planes)
        planes = planes.view(b_size, self.num_planes, self.appearance_features + self.motion_features,
                             planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        # feature_samples, depth_samples, weights_samples = \
        #     self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        rgb_image, peep_video, features, flows_and_masks = self.renderer(planes, self.decoder, c, None,
                                                                         self.rendering_kwargs)  # channels last

        # Reshape into 'raw' image
        H = W = self.neural_rendering_resolution
        rgb_image = rgb_image.permute(0, 2, 1).reshape(b_size, rgb_image.shape[-1], H, W).contiguous()
        feature_image = features.permute(0, 2, 1).reshape(b_size, features.shape[-1], H, W).contiguous()
        # depth_image = depth_samples.permute(0, 2, 1).reshape(b_size, 1, H, W)

        # sr_image = torch.zeros((b_size, rgb_image.shape[1], self.img_resolution, self.img_resolution),
        #                         device=rgb_image.device)
        # xy_idx = (c[:, 2] == 1)
        # rgb_xy_image = rgb_image[xy_idx]
        rgb_xy_image = rgb_image
        #
        # if len(rgb_xy_image) > 0:
        #     feature_xy_image = feature_image[xy_idx]
        #     ws_xy = ws[xy_idx]

        feature_xy_image = feature_image
        ws_xy = ws

        # Run superresolution to get final image
        sr_xy_image = self.superresolution(
            rgb_xy_image, feature_xy_image, ws_xy,
            noise_mode=self.rendering_kwargs['superresolution_noise_mode'],
            **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        sr_image = sr_xy_image

            # sr_image[xy_idx] = sr_xy_image
        # import ipdb; ipdb.set_trace()

        # sr_image = rgb_image
        # rgb_image = self.downscale4x(sr_image)

        return {'image': sr_image, 'image_raw': rgb_image, 'peep_vid': peep_video, 'flows_and_masks': flows_and_masks}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False,
               **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), self.num_planes, self.appearance_features, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False,
                     **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), self.num_planes, self.appearance_features, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
                              cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone,
                              **synthesis_kwargs)


from training.networks_stylegan2 import Conv2dLayer, SynthesisBlock

class ScaleForwardIdentityBackward(torch.autograd.Function):
     @staticmethod
     def forward(ctx, inp, mult_fact):
         result = inp * mult_fact
         # ctx.save_for_backward(result)
         return result

     @staticmethod
     def backward(ctx, grad_output):
         return grad_output, None


class Sin(torch.nn.Module):
    def __init__(self, omega, optimizable_freq=False):
        super().__init__()
        if optimizable_freq:
            self.omega = torch.nn.Parameter(torch.tensor(omega, dtype=torch.float32))
        else:
            self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        # self.rend_res = options['rend_res']

        self.synth_net = torch.nn.ModuleList([
            SynthesisBlock(in_channels=n_features, out_channels=8 * n_features, w_dim=4, resolution=None,
                           img_channels=3, use_noise=False, is_last=False, up=1,
                           activation='lrelu', kernel_size=3, architecture='orig',),

            SynthesisBlock(in_channels=8 * n_features, out_channels=n_features, w_dim=4, resolution=None,
                           img_channels=3, use_noise=False, is_last=False, up=1, kernel_size=1,
                           activation='lrelu', architecture='resnet'),

            # SynthesisBlock(in_channels=n_features, out_channels=n_features, w_dim=4, resolution=None,
            #                img_channels=3, use_noise=False, is_last=False, up=1, activation='relu', kernel_size=1,
            #                architecture='skip'),
            SynthesisBlock(in_channels=n_features, out_channels=options['decoder_output_dim'], w_dim=4, resolution=None,
                           img_channels=3, use_noise=False, is_last=False, up=1,
                           activation='lrelu',
                           kernel_size=1, architecture='skip')])

        # import ipdb; ipdb.set_trace()

        # self.modulator = torch.nn.ModuleList(
        #     [Conv2dLayer(in_channels=n_features * 3, out_channels=4*n_features, kernel_size=1,  activation='relu'),
        #      Conv2dLayer(in_channels=4*n_features, out_channels=n_features, kernel_size=1,  activation='relu'),
        #      Conv2dLayer(in_channels=n_features, out_channels=options['decoder_output_dim'], kernel_size=1,
        #                  activation='relu'),])

    def forward(self, sampled_features, full_rendering_res, bypass_network=False):
        """sampled_features: batch_size, num_pts, pln_chnls"""
        # Aggregate features
        # print(f'feature:{sampled_features[0, :, 100, :4]}')
        # sampled_features = sampled_features.mean(1, keepdim=True)
        rend_cols = full_rendering_res[0]
        time_steps = full_rendering_res[-1]
        batch_size, num_pts, pln_chnls = sampled_features.shape

        ws = torch.zeros((batch_size * time_steps, 3, 4), dtype=sampled_features.dtype,
                         device=sampled_features.device)
        # features received as -> b, num_pts, feature_dim, where -> num_pts = row_cods * col_cods * t_cods
        x = sampled_features.reshape(batch_size, rend_cols, rend_cols, time_steps, pln_chnls)
        # -----------------------------------------------------------------------------------------------
        x = x.permute(0, 3, 1, 2, 4).reshape(batch_size * time_steps, rend_cols, rend_cols, pln_chnls)
        # permuted to b, time_steps, rows, cols, features then reshaped
        # -----------------------------------------------------------------------------------------------
        x = x.permute(0, 3, 1, 2).reshape(batch_size * time_steps, pln_chnls, rend_cols, rend_cols)
        # permuted to b*time_steps, plane_channels, rows, cols
        # -----------------------------------------------------------------------------------------------

            # import ipdb; ipdb.set_trace()
        # x = x * cods.repeat(planes//3, pln_chnls, 1, 1)
        # x = x / 0.23730 * 0.01/4
        # x = x * 0.05952380952380953 / 3
        synth_h = x # * 0.01 / 1.9
        # x = ScaleForwardIdentityBackward.apply(x, 0.005952380952380953)
        # import ipdb; ipdb.set_trace()
        if bypass_network:
            # Jsust for debug purpose
            img = synth_h[:, :3]
        else:
            img = None
            for i, synth_layer in enumerate(self.synth_net):
                synth_layer.resolution = rend_cols
                # mod_layer = self.modulator[i]
                # if i == 2:
                #     skip = x
                # elif i == len(self.net) - 3:
                #     x = x + skip
                # import ipdb; ipdb.set_trace()
                if synth_layer.architecture == 'skip':
                    synth_h, img = synth_layer(synth_h, img, ws)
                else:
                    synth_h, img = synth_layer(synth_h, img, ws[:, :2, :])
                # x = mod_layer(x)
                # synth_h = synth_h * x

        img = img.reshape(batch_size, time_steps, 3, rend_cols, rend_cols)
        # b, time, color, row, col
        img = img.permute(0, 3, 4, 1, 2).reshape(batch_size, num_pts, 3)
        synth_h = synth_h.view(batch_size, time_steps, -1, rend_cols, rend_cols)
        synth_h = synth_h.permute(0, 3, 4, 1, 2).reshape(batch_size, num_pts, -1)

        return {'rgb': img.squeeze(), 'features': synth_h.squeeze()}
