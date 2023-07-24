import torch
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, RotateCamInCirclePerPtoZWhileLookingAt
import numpy as np


device = 'cpu'
cam_pivot = torch.tensor([0, 0, 0], device=device)
cam_radius = 2.7
circle_radiuous = 0.5
rot_angle = 0
fov_deg = 18.837
intrinsics = FOV_to_intrinsics(fov_deg, device=device)

cam2world_pose = RotateCamInCirclePerPtoZWhileLookingAt.sample(
                cam_pivot, z_dist=np.sqrt(cam_radius**2 - circle_radiuous**2), circle_radius=circle_radiuous,
                rot_angle=rot_angle, device=device)
camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
print(camera_params)