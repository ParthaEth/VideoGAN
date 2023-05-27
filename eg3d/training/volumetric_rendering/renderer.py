# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math

import ipdb
import numpy as np
import torch
import torch.nn as nn

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils, mh_projector


class SampleUsingMHA(torch.nn.Module):
    def __init__(self, num_plane_features):
        super().__init__()
        # self.projector = mh_projector.MHprojector(motion_feature_dim=num_plane_features//2,
        #                                           appearance_feature_dim=num_plane_features-num_plane_features//2,
        #                                           num_heads=4)
        self.projector = mh_projector.TransformerProjector(proj_dim=num_plane_features, num_heads=8)
        # self.projector = mh_projector.TinyMHprojector(proj_dim=num_plane_features, num_heads=4)

    def forward(self, plane_features, coordinates, recompute_full_vid_features, bypass_network=False):
        batch_size, n_planes, C, H, W = plane_features.shape
        assert n_planes == 1
        _, num_pts, _ = coordinates.shape

        if bypass_network:
            output_features = coordinates.view(batch_size, 1, num_pts, 3).repeat(1, n_planes, 1, 1)
        else:
            # perform multihead attention on the planer features output : batch_size, n_planes, num_pts, C
            output_features, attention_mask = self.projector(plane_features.squeeze(), coordinates,
                                                             recompute_full_vid_features)
        return output_features[:, None, ...], attention_mask[:, None, ...]


class ImportanceRenderer(torch.nn.Module):
    def __init__(self, plane_feature_dim):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.projector = SampleUsingMHA(plane_feature_dim)

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)


        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        return rgb_final, depth_final, weights.sum(2)

    def run_model(self, planes, decoder, sample_coordinates, options, render_one_frame, recompute_full_vid_features,
                  bypass_network=False):
        sampled_features, attention_mask = self.projector(planes, sample_coordinates, recompute_full_vid_features,
                                                          bypass_network=bypass_network)
        if render_one_frame:
            full_rendering_res = (options['neural_rendering_resolution'],
                                  options['neural_rendering_resolution'],
                                  1)
        else:
            full_rendering_res = (options['neural_rendering_resolution'],
                                  options['neural_rendering_resolution'],
                                  options['time_steps'])

        out = decoder(sampled_features, full_rendering_res, bypass_network=bypass_network)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out, attention_mask

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples


class AxisAligndProjectionRenderer(ImportanceRenderer):
    def __init__(self, return_video, plane_feature_dim):
        # self.neural_rendering_resolution = neural_rendering_resolution
        self.return_video = return_video
        super().__init__(plane_feature_dim)

    def forward(self, planes, decoder, c, ray_directions, rendering_options):
        # assert ray_origins is None  # This will be ignored silently
        # assert ray_directions is None   # This will be ignored silently
        device = planes.device
        datatype = planes.dtype

        batch_size, _, planes_ch, _, _ = planes.shape  # get batch size! ray_origins.shape
        num_coordinates_per_axis = rendering_options['neural_rendering_resolution']
        axis_x = torch.linspace(-1.0, 1.0, num_coordinates_per_axis, dtype=datatype, device=device)
        axis_y = torch.linspace(-1.0, 1.0, num_coordinates_per_axis, dtype=datatype, device=device)
        if self.return_video:  # Remove hack
            # import ipdb; ipdb.set_trace()
            assert(torch.all(-0.01 <= c[:, 3]) and torch.all(c[:, 3] <= 1.01))
            axis_t = c[:, 3] * 2 - 1
            # if not self.training:
            #     assert (torch.all(c[:, 1] < 0.05) and torch.all(c[:, 1] > -0.05))
        else:
            axis_t = torch.zeros(batch_size, dtype=torch.float32, device=device) - 1

        grid_x, grid_y = torch.meshgrid(axis_x, axis_y, indexing='ij')

        sample_coordinates = []
        for b_id in range(batch_size):
            axis_t_this_smpl = axis_t[b_id].repeat(grid_x.shape)

            if torch.argmax(c[b_id, 0:3]) == 0:
                coordinates = [axis_t_this_smpl[None, ...], grid_x[None, ...], grid_y[None, ...]]
                # print(f'problem {c[b_id, 0:2]}')
                # print(f'x: {axis_t}')
            elif torch.argmax(c[b_id, 0:3]) == 1:
                coordinates = [grid_x[None, ...], axis_t_this_smpl[None, ...], grid_y[None, ...]]
                # print(f'problem {c[b_id, 0:2]}')
                # print(f'y: {axis_t}')
            elif torch.argmax(c[b_id, 0:3]) == 2:
                coordinates = [grid_x[None, ...], grid_y[None, ...], axis_t_this_smpl[None, ...]]
                # print(c[0, 0:2], axis_t_this_smpl[0, 0])
                # print(f't: {axis_t}')
            else:
                raise ValueError(f'Constant axis index must be between 0 and 2 got {int(c[0, 0])}')
            # if self.training and self.return_video:  # In eval mode we sample pixel with random but constant time label
            #     random.shuffle(coordinates)
                # print('In render.py. Shuffling the axes')

            sample_coordinates += torch.stack(coordinates, dim=3)

        # import ipdb; ipdb.set_trace()
        sample_coordinates = torch.stack(sample_coordinates, dim=0)\
            .reshape((batch_size, num_coordinates_per_axis*num_coordinates_per_axis, 3))
        # sample_coordinates = sample_coordinates + torch.randn_like(sample_coordinates)/100
        # print(f'coord: {sample_coordinates[0, :2, :]}')
        # sample_directions = sample_coordinates

        out, img_attn_mask = self.run_model(planes, decoder, sample_coordinates, rendering_options,
                                            recompute_full_vid_features=True, render_one_frame=True)
        colors_coarse, features = out['rgb'], out['features']

        # render peep video
        norm_peep_cod = c[:, 4:6] * 2 - 1
        assert (torch.all(-1.01 <= norm_peep_cod) and torch.all(norm_peep_cod + 2/4 <= 1.01))
        video_coordinates = []
        video_spatial_res = num_coordinates_per_axis // 2
        vide_time_res = rendering_options['time_steps']
        for b_id in range(batch_size):
            cod_x = torch.linspace(norm_peep_cod[b_id, 0], norm_peep_cod[b_id, 0] + 2/4,
                                   video_spatial_res, dtype=datatype, device=device)
            cod_y = torch.linspace(norm_peep_cod[b_id, 1], norm_peep_cod[b_id, 1] + 2/4,
                                   video_spatial_res, dtype=datatype, device=device)
            cod_z = torch.linspace(-1, 1, vide_time_res, dtype=datatype, device=device)
            grid_x, grid_y, grid_z = torch.meshgrid(cod_x, cod_y, cod_z, indexing='ij')
            coordinates = torch.stack((grid_x, grid_y, grid_z), dim=0).permute(1, 2, 3, 0)
            video_coordinates.append(coordinates)

        # import ipdb; ipdb.set_trace()
        video_coordinates = torch.stack(video_coordinates, dim=0).reshape(batch_size, -1, 3)
        rendering_options['neural_rendering_resolution'] = video_spatial_res
        # rendering_options['time_steps'] = vide_time_res
        out, vid_attn_mask = self.run_model(planes, decoder, video_coordinates, rendering_options,
                                            recompute_full_vid_features=True, render_one_frame=False)
        peep_vid = out['rgb'].reshape(batch_size, video_spatial_res, video_spatial_res, vide_time_res, 3)\
            .permute(0, 4, 1, 2, 3)  # b, color, x, y, t

        return colors_coarse, peep_vid, features, img_attn_mask
