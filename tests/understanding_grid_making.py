import torch
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image

def get_2D_grid(rend_res, dtype, device):
    cod_x = cod_y = torch.linspace(-1, 1, rend_res, dtype=dtype, device=device)
    grid_x, grid_y = torch.meshgrid(cod_x, cod_y, indexing='ij')
    coordinates = torch.stack((grid_x, grid_y), dim=0).permute(1, 2, 0)
    # coordinates = coordinates.reshape(1, -1, 2).expand(batch, -1, -1)
    return coordinates


def forward_warp(feature_frame, grid):
    return torch.nn.functional.grid_sample(feature_frame, grid, align_corners=True, padding_mode='border')

def prepare_feature_volume(planes, grid):
    """Plane: a torch tensor b, 1, 38, h, w"""
    # rend_res = options['neural_rendering_resolution']
    # 5 =((w, h), (w, h), mask)
    global_appearance_features = planes

    appearance_volume = []
    prev_frame = None
    for time_id in range(64):
        global_features = forward_warp(global_appearance_features.permute(0, 1, 3, 2), grid)
        # global_features: batch, ch, h, w notice the assumed convention of lf_gfc_mask
        if prev_frame is None:
            prev_frame = current_frame = global_features
        else:
            fwd_frm = forward_warp(prev_frame.permute(0, 1, 3, 2), grid)
            prev_frame = current_frame = fwd_frm

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(current_frame.permute(0, 2, 3, 1)[0].cpu())
        ax2.imshow(planes.permute(0, 2, 3, 1)[0].cpu())
        flow = get_2D_grid(grid.shape[1], grid.dtype, grid.device)[None] - grid
        local_flow = flow_to_image(flow[0].permute(2, 0, 1))
        ax3.imshow(local_flow.permute(1, 2, 0).cpu())
        plt.show()
        appearance_volume.append(current_frame)

    # appearance_volume = torch.stack(appearance_volume, dim=0)  # shape: t, batch, app_feat, h, w


grid = get_2D_grid(64, torch.float32, 'cuda')
input = torch.zeros(3, 64, 64, device='cuda')
input[:, 5:40, 5:20] = 1.0
input[:, :, [0, -1]] = 0.5
input[:, [0, -1], :] = 0.5
input[:, 32, :] = 0.5
input[:, :, 32] = 0.5
grid[32:, 32:, 0] -= 0.1
grid[32:, 32:, 1] -= 0.2
prepare_feature_volume(input[None, ...], grid[None, ...])