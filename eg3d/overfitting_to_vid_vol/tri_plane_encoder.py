import torch
from einops import rearrange
import numpy as np


class InterpolationLayer(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size

    def forward(self, inp):
        """

        Args:
            inp: b, ch, rows, cols, ...

        Returns:
            interpolated output with sie b, ch, out_size[0], out_size[1], ...
        """
        return torch.nn.functional.interpolate(inp, self.out_size)


class TriplaneEncoder(torch.nn.Module):
    def __init__(self, input_vid_res, tri_plane_res):
        super().__init__()
        """
        input_vid_res: must be a 4 tuple of ch, row_res, col_res, depth res
        tri_plane_res: must be a 3 tuple of ch, row_res, col_res
        """
        self.tri_plane_res = tri_plane_res
        self.input_vid_res = input_vid_res
        self.enc_res_preserve_rc = torch.nn.Sequential(torch.nn.Linear(input_vid_res[3] * input_vid_res[0],
                                                                       tri_plane_res[0]),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(tri_plane_res[0], tri_plane_res[0]),
                                                       )
        self.enc_res_preserve_cd = torch.nn.Sequential(torch.nn.Linear(input_vid_res[1] * input_vid_res[0],
                                                                       tri_plane_res[0]),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(tri_plane_res[0], tri_plane_res[0]),
                                                       )
        self.enc_res_preserve_rd = torch.nn.Sequential(torch.nn.Linear(input_vid_res[2] * input_vid_res[0],
                                                                       tri_plane_res[0]),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(tri_plane_res[0], tri_plane_res[0]),
                                                       )

        conv_layers = 2
        down_scale_resolutions = np.linspace(input_vid_res[1], tri_plane_res[1], conv_layers + 1)[1:].astype(int)
        self.enc_conv_down = torch.nn.Sequential(torch.nn.Conv2d(3*tri_plane_res[0], 3*tri_plane_res[0], kernel_size=3,
                                                                 padding=1),
                                                 InterpolationLayer(down_scale_resolutions[0]),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Conv2d(3 * tri_plane_res[0], tri_plane_res[0],
                                                                 kernel_size=3,
                                                                 padding=1),
                                                 InterpolationLayer(down_scale_resolutions[1])
                                                 )

    def forward(self, vid_vol):
        """
        Args:
            vid_vol: batch, ch, rows, cols, depth

        Returns:
            triplane: batch, ch, rows, cols, 3
        """
        plane_row_col = self.enc_res_preserve_rc(rearrange(vid_vol, 'b ch row col dep -> b row col (dep ch)'))
        plane_col_dep = self.enc_res_preserve_cd(rearrange(vid_vol, 'b ch row col dep -> b col dep (row ch)'))
        plane_row_dep = self.enc_res_preserve_rd(rearrange(vid_vol, 'b ch row col dep -> b row dep (col ch)'))
        planes = torch.cat([plane_row_col, plane_col_dep, plane_row_dep], dim=-1)  # shape = b, row, col, 3*feat_dim
        planes = rearrange(planes, 'b row col fd -> b fd row col')
        planes = self.enc_conv_down(planes)

        triplane = rearrange(planes, 'b (n_p fd) row col -> b n_p fd row col', n_p=1)
        return triplane


if __name__ == '__main__':
    input_vid_res = 3, 32, 32, 32
    tri_plane_res = 8, 16, 16
    enc = TriplaneEncoder(input_vid_res, tri_plane_res)
    vid = torch.randn([1, ] + list(input_vid_res))

    triplanes = enc(vid)
    print(triplanes.shape)
