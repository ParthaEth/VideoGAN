# import sys
# sys.path.append('../eg3d')
# from torch_utils import misc, persistence
# import dataclasses
# @persistence.persistent_class
# @dataclasses.dataclass(eq=False, repr=False)
# class Foo():
#     i:int = 1
#
#     def __post_init__(self):
#         super().__init__()
#
#
# print('Before __init__')
# foo = Foo(5)
# print('After __init__')
# print(foo.i)
import numpy as np
import torch
batch_size = 3
from eg3d.training.volumetric_rendering.renderer import AxisAligndProjectionRenderer
from eg3d.training.triplane import OSGDecoder

norm_peep_cod = np.array([[-1, -1], [-1, -0.5], [0.5, -1]])
video_spatial_res = 4
datatype = torch.float32
device = 'cpu'
vide_time_res = 1
video_coordinates = []
renderer = AxisAligndProjectionRenderer(return_video=True, num_planes=3)
decoder = OSGDecoder(32, {'decoder_lr_mul': 1, 'decoder_output_dim': 32}).to(device)
options = {}
options['neural_rendering_resolution'] = 4
options['time_steps'] = 1
options['box_warp'] = 1

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
planes = torch.zeros((batch_size, 6, 32, 64, 64))
rend_cods = renderer.run_model(planes=planes, decoder=decoder, sample_coordinates=video_coordinates, options=options,
                               bypass_network=True)
peep_vid = rend_cods['rgb'].reshape(batch_size, video_spatial_res, video_spatial_res, vide_time_res, 3)\
            .permute(0, 4, 1, 2, 3)  # b, color, x, y, t
peep_vid[0,:2, :, :, 0]
peep_vid_expt = video_coordinates.reshape(batch_size, video_spatial_res, video_spatial_res, vide_time_res, 3)\
            .permute(0, 4, 1, 2, 3)   * 2# b, color, x, y, t
peep_vid_expt[0,:2, :, :, 0]
a = 0