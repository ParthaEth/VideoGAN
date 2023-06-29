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

import torch


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],  # XY
                          [0, 1, 0],
                          [0, 0, 1]],

                         [[1, 0, 0],  # XZ
                          [0, 0, 1],
                          [0, 1, 0]],

                         [[0, 0, 1],  # ZY
                          [0, 1, 0],
                          [1, 0, 0]]], dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros'):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(N*n_planes, C, H, W)

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    # import ipdb; ipdb.set_trace()
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode,
                                                      padding_mode=padding_mode, align_corners=False)
    output_features = output_features.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features


def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


class BaseRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.plane_axes = torch.nn.Parameter(generate_planes(num_planes))
        self.register_buffer('plane_axes', generate_planes())

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):
        raise NotImplementedError()


class AxisAligndProjectionRenderer(BaseRenderer):
    def __init__(self, return_video):
        # self.neural_rendering_resolution = neural_rendering_resolution
        self.return_video = return_video
        super().__init__()
        self.appearance_volume = None
        self.lf_gfc_mask = None

    def forward_warp(self, feature_frame, grid):
        return torch.nn.functional.grid_sample(feature_frame, grid, align_corners=True)

    def get_rgb_given_coordinate(self, planes, decoder, sample_coordinates, options, bypass_network=False):
        batch, n_pt, dims = sample_coordinates.shape
        assert dims == 3

        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros')

        # sampled_features is of shape batch, planes, num_pts, pln_chnls; with planes == 3
        sampled_features = sampled_features.permute(0, 2, 1, 3)  # batch, num_pts, planes, pln_ch
        sampled_features = sampled_features.reshape(batch, n_pt, -1)  # batch, num_pts, planes * pln_ch

        out = decoder(sampled_features,
                      full_rendering_res=(options['neural_rendering_resolution'],
                                          options['neural_rendering_resolution'],
                                          options['time_steps']),
                      bypass_network=bypass_network)
        if options.get('density_noise', 0) > 0:
            raise NotImplementedError('This feature has been removed')
        return out

    def forward(self, planes, decoder, c, ray_directions, rendering_options):
        # assert ray_origins is None  # This will be ignored silently
        # assert ray_directions is None   # This will be ignored silently
        device = planes.device
        datatype = planes.dtype

        batch_size, num_planes, planes_ch, _, _ = planes.shape

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

        rendering_options['time_steps'] = 1
        out = self.get_rgb_given_coordinate(planes, decoder, sample_coordinates, rendering_options)
        colors_coarse, features = out['rgb'], out['features']

        # render peep video
        norm_peep_cod = c[:, 4:6] * 2 - 1
        assert (torch.all(-1.01 <= norm_peep_cod) and torch.all(norm_peep_cod + 2/4 <= 1.01))
        video_coordinates = []
        # video_spatial_res = num_coordinates_per_axis // 2
        video_spatial_res = 64
        vide_time_res = 32   # TODO(Partha): pass this coordinate
        for b_id in range(batch_size):
            cod_x = torch.linspace(norm_peep_cod[b_id, 0], norm_peep_cod[b_id, 0] + 2,
                                   video_spatial_res, dtype=datatype, device=device)
            cod_y = torch.linspace(norm_peep_cod[b_id, 1], norm_peep_cod[b_id, 1] + 2,
                                   video_spatial_res, dtype=datatype, device=device)
            cod_z = torch.linspace(-1, 1, vide_time_res, dtype=datatype, device=device)
            grid_x, grid_y, grid_z = torch.meshgrid(cod_x, cod_y, cod_z, indexing='ij')
            coordinates = torch.stack((grid_x, grid_y, grid_z), dim=0).permute(1, 2, 3, 0)
            video_coordinates.append(coordinates)

        # import ipdb; ipdb.set_trace()
        video_coordinates = torch.stack(video_coordinates, dim=0).reshape(batch_size, -1, 3)
        rendering_options['neural_rendering_resolution'] = video_spatial_res
        rendering_options['time_steps'] = vide_time_res
        out = self.get_rgb_given_coordinate(planes, decoder, video_coordinates, rendering_options)
        peep_vid = out['rgb'].reshape(batch_size, video_spatial_res, video_spatial_res, vide_time_res, 3)\
            .permute(0, 4, 1, 2, 3)  # b, color, x, y, t

        return colors_coarse, peep_vid, features
