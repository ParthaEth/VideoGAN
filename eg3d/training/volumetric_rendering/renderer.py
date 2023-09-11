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


def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(N*n_planes, C, H, W)

    # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

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
    def __init__(self, motion_features, appearance_features):
        super().__init__()
        # self.plane_axes = torch.nn.Parameter(generate_planes(num_planes))
        self.register_buffer('plane_axes', generate_planes())
        self.motion_features = motion_features
        self.appearance_features = appearance_features

        self.motion_encoder = torch.nn.Sequential(
            torch.nn.Linear(motion_features, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5, bias=True),
            torch.nn.Tanh()
        )
        #TODO(Partha): Perhaps init weight had significance
        # self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, torch.nn.Linear):
    #         torch.nn.init.trunc_normal_(module.weight.data, mean=0, std=0.15, a=-1, b=1)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

        self.pos_emb = None
    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):
        raise NotImplementedError()

    def get_3D_grid(self, rend_res, dtype, device):
        """
        Args:
            rend_res: int
            dtype: float32/ any other
            device: cuda/cpu

        Returns:
            coordinates: of shape (rend_res, rend_res, rend_res, 3)
        """
        cod_x = cod_y = cod_z = torch.linspace(-1, 1, rend_res, dtype=dtype, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(cod_x, cod_y, cod_z, indexing='ij')
        coordinates = torch.stack((grid_x, grid_y, grid_z), dim=0).permute(1, 2, 3, 0)
        return coordinates

    def get_2D_grid(self, rend_res, dtype, device):
        cod_x = cod_y = torch.linspace(-1, 1, rend_res, dtype=dtype, device=device)
        grid_x, grid_y = torch.meshgrid(cod_x, cod_y, indexing='ij')
        coordinates = torch.stack((grid_x, grid_y), dim=0).permute(1, 2, 0)
        # coordinates = coordinates.reshape(1, -1, 2).expand(batch, -1, -1)
        return coordinates

    def get_identity_flow_and_mask(self, flow_and_mask_shape, dtype, device):
        assert flow_and_mask_shape[0] == flow_and_mask_shape[1]
        identity_lf = self.get_2D_grid(flow_and_mask_shape[0], dtype, device)[None, ...]
        identity_gfc = identity_lf
        identity_mask = torch.ones(flow_and_mask_shape, dtype=dtype, device=device)[None, ..., None] * 0.5
        identity_lf_gfc_mask = torch.cat((identity_lf, identity_gfc, identity_mask), dim=-1)
        return identity_lf_gfc_mask

    def get_motion_feature_vol(self, feature_grid, options):
        """ feature_grid:
                Planes: batch_size, n_planes, channels, h, w ; n_planes == 3 is assumed later
                voxels: batch_size, channels, h, w, d
            return lf_gfc_mask: batch, rend_res, rend_res, rend_res, 5
        """

        batch, _, _, _, _ = feature_grid.shape
        rend_res = options['neural_rendering_resolution']
        feature_grid_type = options['feature_grid_type']
        dtype, device = feature_grid.dtype, feature_grid.device
        coordinates = self.get_3D_grid(rend_res, dtype, device).reshape(1, -1, 3).expand(batch, -1, -1)
        if feature_grid_type.lower() == 'triplane':
            lf_gfc_mask = sample_from_planes(self.plane_axes, feature_grid, coordinates, padding_mode='zeros',
                                             box_warp=options['box_warp'])
            # lf_gfc_mask is of shape batch, planes, num_pts, pln_chnls; with planes == 3
            # import ipdb; ipdb.set_trace()
            lf_gfc_mask = lf_gfc_mask.permute(0, 2, 1, 3)  # lf_gfc_mask is of shape batch, num_pts, planes, pln_chnls
            # lf_gfc_mask is of shape batch, num_pts, planes*pln_chnls
        elif feature_grid_type.lower() == '3d_voxels' or 'positional_embedding':
            # feature_grid of shape batch, channel, height, width, depth
            # coordinates shape batch, h_c, w_c, d_c, 3
            lf_gfc_mask = torch.nn.functional.grid_sample(
                feature_grid, coordinates[:, None, None, ...], align_corners=True)
            # lf_gfc_mask shape: batch, motion_features, 1, 1, N_pt
            lf_gfc_mask = lf_gfc_mask.squeeze(1, 2, 3, 4).permute(0, 2, 1)  # batch, n_pt, motion_features
        else:
            raise ValueError(f'{feature_grid_type} not understood')

        lf_gfc_mask = lf_gfc_mask.reshape(batch, rend_res, rend_res, rend_res, self.motion_features)
        # transposing x and y as torch gris type'ij' transposes the input
        lf_gfc_mask = lf_gfc_mask.permute(0, 2, 1, 3, 4)  # batch, height, width, depth, ch

        identity_flow_and_mask = self.get_identity_flow_and_mask((rend_res, rend_res), dtype, device)
        identity_flow_and_mask = identity_flow_and_mask[:, :, :, None, :].expand(batch, -1, -1, rend_res, -1)
        # import ipdb; ipdb.set_trace()
        generated_flow_mask = self.motion_encoder(lf_gfc_mask).clone()

        # small Global flow ensured
        generated_flow_mask[:, :, :, :, 2:4] = \
            generated_flow_mask[:, :, :, :, 2:4]/16 + identity_flow_and_mask[:, :, :, :, 2:4]  # just keeping flow low
        # mask is between -1 and 1, it will be later normalized
        # small-er local flow ensured
        generated_flow_mask[:, :, :, :, :2] = \
            generated_flow_mask[:, :, :, :, :2] / 64 + identity_flow_and_mask[:, :, :, :, :2]  # just keeping flow low
        return generated_flow_mask  # batch, rend_res, rend_res, rend_res, 5


class AxisAligndProjectionRenderer(BaseRenderer):
    def __init__(self, return_video, motion_features, appearance_features):
        # self.neural_rendering_resolution = neural_rendering_resolution
        self.return_video = return_video
        super().__init__(motion_features, appearance_features)
        self.appearance_volume = None
        self.lf_gfc_mask = None

    def forward_warp(self, feature_frame, grid):
        return torch.nn.functional.grid_sample(feature_frame, grid, align_corners=True)

    def prepare_feature_volume(self, feature_grid, options, bypass_network=False):
        """Plane: a torch tensor b, 1, 38, h, w"""
        rend_res = options['neural_rendering_resolution']
        # fist self.motion_features are assumed to be motion features
        feature_grid_type = options['feature_grid_type']
        if feature_grid_type.lower() == 'triplane':
            batch, n_planes, channels, h, w = feature_grid.shape
            assert n_planes == 1, 'Here it is assumed that planes are stacked in the channel dims'
            motion_feature_grid = feature_grid[:, :, :self.motion_features, :, :].reshape(batch, 3, -1, h, w)
            global_appearance_features = feature_grid[:, 0, self.motion_features:, :, :]
        elif feature_grid_type.lower() == '3d_voxels' or 'positional_embedding':
            # feature_grid shape: batch, chn, h, w, d
            # motion_feature_grid shape: batch, motion_features, h, w, d
            motion_feature_grid = feature_grid[:, :self.motion_features, :, :, :]

            # First plane is assumed to be global appearance features
            # global_features: batch, ch, h, w
            global_appearance_features = feature_grid[:, self.motion_features:, :, :, 0]
        else:
            raise ValueError(f'{feature_grid_type} not understood')

        self.lf_gfc_mask = self.get_motion_feature_vol(motion_feature_grid, options)
        # self.lf_gfc_mask is of shape batch, rend_res (height), rend_res (width), rend_res (depth),
        # 5 =((w, h), (w, h), mask)

        appearance_volume = []
        prev_frame = None
        for time_id in range(rend_res):
            global_features = self.forward_warp(global_appearance_features.permute(0, 1, 3, 2),
                                                self.lf_gfc_mask[:, :, :, time_id, 2:4])
            # global_features: batch, ch, h, w notice the assumed convention of lf_gfc_mask
            if prev_frame is None:
                prev_frame = current_frame = global_features
            else:
                # prev_frame must be permuted or else we will be transposing it all the time
                fwd_frm = self.forward_warp(prev_frame.permute(0, 1, 3, 2), self.lf_gfc_mask[:, :, :, time_id, :2])
                # import ipdb; ipdb.set_trace()
                # fwd_frm: batch, ch, h, w notice the assumed convention of lf_gfc_mask
                mask = (self.lf_gfc_mask[:, :, :, time_id, 4] + 1) / 2
                prev_frame = current_frame = global_features * mask[:, None, ...] + fwd_frm * (1 - mask[:, None, ...])

            appearance_volume.append(current_frame)

        appearance_volume = torch.stack(appearance_volume, dim=0)  # shape: t, batch, app_feat, h, w
        self.appearance_volume = appearance_volume.permute(1, 2, 0, 3, 4)  # shape: batch, app_feat, t, h, w

    def get_rgb_given_coordinate(self, decoder, sample_coordinates, options, bypass_network=False):
        batch, n_pt, dims = sample_coordinates.shape
        assert dims == 3
        sample_coordinates = sample_coordinates.reshape(batch, 1, 1, n_pt, dims)  # dims := (h, w, t)
        # self.appearance_volume: batch, app_feat, t, h, w
        # Sample cods are flipping h and w in the following grid sample. swap appearnace feature h, w dims?
        sampled_features = torch.nn.functional.grid_sample(self.appearance_volume, sample_coordinates,
                                                           align_corners=True, padding_mode='border').squeeze()
        # sampled_features := batch, ch, 1, 1, n_pt -> squeez -> batch, ch, n_pt
        sampled_features = sampled_features.permute(0, 2, 1)  # batch, num_pts, features
        # import ipdb; ipdb.set_trace()

        # self.lf_gfc_mask is of shape batch, rend_res (height), rend_res (width), rend_res (depth=t),
        # 5 =((w, h), (w, h), mask), # sample_coordinates: (batch, npts, dims (dims := h, w, t))
        # But grid sample assumes sample cods to have depth, with, height ordering so the permutation
        flows_and_mask = torch.nn.functional.grid_sample(self.lf_gfc_mask.permute(0, 4, 3, 1, 2), sample_coordinates,
                                                         align_corners=True, padding_mode='border').squeeze()
        out = decoder(sampled_features,
                      full_rendering_res=(options['neural_rendering_resolution'],
                                          options['neural_rendering_resolution'],
                                          options['time_steps']),
                      bypass_network=bypass_network)
        if options.get('density_noise', 0) > 0:
            raise NotImplementedError('This feature has been removed')
        out['flows_and_mask'] = flows_and_mask
        return out

    def forward(self, feature_grid, decoder, c, ray_directions, rendering_options):
        # assert ray_origins is None  # This will be ignored silently
        # assert ray_directions is None   # This will be ignored silently
        device = feature_grid.device
        datatype = feature_grid.dtype
        # self.plane_axes = self.plane_axes.to(device)
        self.prepare_feature_volume(feature_grid, rendering_options, bypass_network=False)

        if rendering_options['feature_grid_type'].lower() == 'triplane':
            batch_size, num_planes, _, _, _ = feature_grid.shape
            assert num_planes == 1, 'right now appearance planes and 3 flow planes are all in channel dim'
        else:
            batch_size, _, _, _, _ = feature_grid.shape

        num_coordinates_per_axis = rendering_options['neural_rendering_resolution']
        axis_x = torch.linspace(-1.0, 1.0, num_coordinates_per_axis, dtype=datatype, device=device) * 0.5
        axis_y = torch.linspace(-1.0, 1.0, num_coordinates_per_axis, dtype=datatype, device=device) * 0.5
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
        out = self.get_rgb_given_coordinate(decoder, sample_coordinates, rendering_options)
        colors_coarse, features, flows_and_mask = out['rgb'], out['features'], out['flows_and_mask']

        # render peep video
        if c.shape[-1] >= 6:
            norm_peep_cod = c[:, 4:6] * 2 - 1
            assert (torch.all(-1.01 <= norm_peep_cod) and torch.all(norm_peep_cod + 2/4 <= 1.01))
            video_coordinates = []
            video_spatial_res = num_coordinates_per_axis // 2
            vide_time_res = 32   # TODO(Partha): pass this coordinate
            for b_id in range(batch_size):
                cod_x = torch.linspace(norm_peep_cod[b_id, 0], norm_peep_cod[b_id, 0] + 2,
                                       video_spatial_res, dtype=datatype, device=device)
                cod_y = torch.linspace(norm_peep_cod[b_id, 1], norm_peep_cod[b_id, 1] + 2,
                                       video_spatial_res, dtype=datatype, device=device)
                cod_z = torch.linspace(-1, 1, vide_time_res, dtype=datatype, device=device)
                grid_x, grid_y, grid_z = torch.meshgrid(cod_x * 0.5, cod_y * 0.5, cod_z, indexing='ij')
                coordinates = torch.stack((grid_x, grid_y, grid_z), dim=0).permute(1, 2, 3, 0)
                video_coordinates.append(coordinates)

            # import ipdb; ipdb.set_trace()
            video_coordinates = torch.stack(video_coordinates, dim=0).reshape(batch_size, -1, 3)
            rendering_options['neural_rendering_resolution'] = video_spatial_res
            rendering_options['time_steps'] = vide_time_res
            out = self.get_rgb_given_coordinate(decoder, video_coordinates, rendering_options)
            peep_vid = out['rgb'].reshape(batch_size, video_spatial_res, video_spatial_res, vide_time_res, 3)\
                .permute(0, 4, 1, 2, 3)  # b, color, x, y, t
        else:
            peep_vid = None

        return colors_coarse, peep_vid, features, flows_and_mask
