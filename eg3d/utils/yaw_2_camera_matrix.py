from PIL import Image
import os
import tqdm
import json
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import numpy as np
import torch
import math

input_json_path = '/home/pghosh/Downloads/dataset_2(1).json'
dest_dir = '/home/pghosh/Downloads/'
dataset_description = os.path.join(dest_dir, 'dataset_2.json')
file_object = open(dataset_description, 'w')
file_object.write('{\n')
file_object.write('    "labels": [\n')

with open(input_json_path) as f:
    labels = json.load(f)['labels']
labels = dict(labels)
num_files = len(labels)
curr_count = 0
max_count = 9999999
device = 'cpu'
cam_pivot = torch.tensor([0, 0, 0], device=device)
angle_p = -0.2
cam_radius = 2.7
fov_deg = 18.837
intrinsics = FOV_to_intrinsics(fov_deg, device=device)
for curr_file in tqdm.tqdm(labels):
    # if curr_file.endswith('.jpg') or curr_file.endswith('.png'):
    curr_count += 1
    angle_y = float(labels[curr_file]) * np.pi / 180
    if math.isnan(angle_y):
        angle_y = 0
    cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius,
                                              device=device)

    camera_param = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).numpy()[0]
    camera_param_str = '['
    for param in camera_param:
        camera_param_str += f'{param}, '
    camera_param_str = camera_param_str[:-2] + ']'
    line_this_img = f'        ["{curr_file}",{camera_param_str}]'
    # import ipdb; ipdb.set_trace()
    if curr_count == num_files or curr_count == max_count:
        line_this_img += '\n'
        print('End of file!')
    else:
        line_this_img += ',\n'

    file_object.write(line_this_img)

    if curr_count >= max_count:
        break

file_object.write('    ]\n')
file_object.write('}\n')
file_object.close()