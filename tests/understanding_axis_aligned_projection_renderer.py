import torch
from training.volumetric_rendering.renderer import AxisAligndProjectionRenderer


class IdentityDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, full_rendering_res, bypass_network=False):
        out = {'rgb': inp, 'features': None}
        return out


axpr = AxisAligndProjectionRenderer(return_video=True)
decoder = IdentityDecoder()
batch_size, num_planes, planes_ch, plane_h, plane_w = 1, 3, 1, 256, 256
planes = torch.rand(batch_size, num_planes, planes_ch, plane_h, plane_w)
timesteps = (torch.linspace(-1, 1, 32) + 1) / 2
image_time_step = 2
c = torch.tensor([[0, 0, 1, timesteps[image_time_step].data, 0, 0]])
ray_directions = None
rendering_options = {'neural_rendering_resolution': 32, }
colors_coarse, peep_vid, features = axpr(planes, decoder, c, ray_directions, rendering_options)

img = colors_coarse.reshape(batch_size, rendering_options['neural_rendering_resolution'],
                            rendering_options['neural_rendering_resolution'], -1).permute(0, 3, 1, 2)

vid_frame_at_time = peep_vid[:, :, :, :, image_time_step]

print((img - vid_frame_at_time).abs().sum())