import torch
import torchvision
import matplotlib.pyplot as plt


def get_identity_flow(rend_res, dtype, device):
    cod_x = cod_y = torch.linspace(-1, 1, rend_res, dtype=dtype, device=device)
    grid_x, grid_y = torch.meshgrid(cod_x, cod_y, indexing='ij')
    coordinates = torch.stack((grid_x, grid_y), dim=0)
    return coordinates


class FakeRenderer:
    def __init__(self):
        self.return_video = True
        identity_grid = get_identity_flow(64, device='cpu', dtype=torch.float32)
        id_flow_img = torchvision.utils.flow_to_image(identity_grid[None]).to(torch.float32) /255
        self.appearance_volume = []
        dark_ness = torch.linspace(1, 8, 64)
        for i in range(64):
            self.appearance_volume.append(id_flow_img/dark_ness[i])

        self.appearance_volume = torch.stack(self.appearance_volume, dim=4)  # batch==1, ch, h, w, t
        self.appearance_volume = self.appearance_volume.permute(0, 1, 4, 3, 2)  # batch==1, app_feat, t, h, w

    def get_rgb_given_coordinate(self, sample_coordinates, options, bypass_network=False):
        batch, n_pt, dims = sample_coordinates.shape
        assert dims == 3
        sample_coordinates = sample_coordinates.reshape(batch, 1, 1, n_pt, dims)  # dims := (h, w, t)
        # self.appearance_volume: batch, app_feat, t, h, w
        # TODO(): Check id the grid sample did the right job
        sampled_features = torch.nn.functional.grid_sample(self.appearance_volume, sample_coordinates,
                                                           align_corners=True, padding_mode='border').squeeze()

        return sampled_features  # returns batch, 3, n_pt

    def forward(self, c, rendering_options):
        # assert ray_origins is None  # This will be ignored silently
        # assert ray_directions is None   # This will be ignored silently
        device = 'cpu'
        datatype = torch.float32

        batch_size = 1
        num_planes = 1

        assert num_planes == 1, 'right now appearance planes and 3 flow planes are all in channel dim'
        num_coordinates_per_axis = rendering_options['neural_rendering_resolution']
        axis_x = torch.linspace(-1.0, 1.0, num_coordinates_per_axis, dtype=datatype, device=device)
        axis_y = torch.linspace(-1.0, 1.0, num_coordinates_per_axis, dtype=datatype, device=device)
        if self.return_video:  # Remove hack
            # import ipdb; ipdb.set_trace()
            assert (torch.all(-0.01 <= c[:, 3]) and torch.all(c[:, 3] <= 1.01))
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
        sample_coordinates = torch.stack(sample_coordinates, dim=0) \
            .reshape((batch_size, num_coordinates_per_axis * num_coordinates_per_axis, 3))
        # sample_coordinates = sample_coordinates + torch.randn_like(sample_coordinates)/100
        # print(f'coord: {sample_coordinates[0, :2, :]}')
        # sample_directions = sample_coordinates

        rendering_options['time_steps'] = 1
        out = self.get_rgb_given_coordinate(sample_coordinates, rendering_options)
        plt.imshow(out.reshape(3, 64, 64).permute(1, 2, 0))
        # plt.imshow(self.appearance_volume[0, :, 0, :, :].permute(1, 2, 0))
        plt.show()


        # render peep video
        norm_peep_cod = c[:, 4:6] * 2 - 1
        assert (torch.all(-1.01 <= norm_peep_cod) and torch.all(norm_peep_cod + 2 / 4 <= 1.01))
        video_coordinates = []
        video_spatial_res = num_coordinates_per_axis // 2
        vide_time_res = num_coordinates_per_axis * 4
        for b_id in range(batch_size):
            cod_x = torch.linspace(norm_peep_cod[b_id, 0], norm_peep_cod[b_id, 0] + 2 / 4,
                                   video_spatial_res, dtype=datatype, device=device)
            cod_y = torch.linspace(norm_peep_cod[b_id, 1], norm_peep_cod[b_id, 1] + 2 / 4,
                                   video_spatial_res, dtype=datatype, device=device)
            cod_z = torch.linspace(-1, 1, vide_time_res, dtype=datatype, device=device)
            grid_x, grid_y, grid_z = torch.meshgrid(cod_x, cod_y, cod_z, indexing='ij')
            coordinates = torch.stack((grid_x, grid_y, grid_z), dim=0).permute(1, 2, 3, 0)
            video_coordinates.append(coordinates)

        # import ipdb; ipdb.set_trace()
        video_coordinates = torch.stack(video_coordinates, dim=0).reshape(batch_size, -1, 3)
        rendering_options['neural_rendering_resolution'] = video_spatial_res
        rendering_options['time_steps'] = vide_time_res
        out = self.get_rgb_given_coordinate(video_coordinates, rendering_options)

        plt.imshow(out.reshape(3, 32, 32, 256)[:, :, :, 0].permute(1, 2, 0))
        plt.show()


if __name__ == '__main__':
    fake_rend = FakeRenderer()
    rendering_options = {'neural_rendering_resolution': 64}
    time_cod = 0.0
    peep_cod = [0.375, 0.375]
    c = torch.tensor([[0, 0, 1, time_cod,] + peep_cod])
    fake_rend.forward(c, rendering_options)
