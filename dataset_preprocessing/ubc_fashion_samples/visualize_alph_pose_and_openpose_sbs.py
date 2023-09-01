import os
import json
import numpy as np
import matplotlib.pyplot as plt

root_dir = '/is/cluster/fast/scratch/ssanyal/video_gan/fashion_videos/match_to_alpha_pose/'

frames = sorted(os.listdir(os.path.join(root_dir, 'frames')))

for frame in frames:
    given_kps_file = frame.replace('.png', '.json')
    with open(os.path.join(root_dir, 'kptsmpls', given_kps_file)) as f:
        given_kps = np.array(json.load(f)['pose_keypoints_2d'])

    with open(os.path.join(root_dir, 'alph_pose_out/sep-json', given_kps_file)) as f:
        alphapose_kps = json.load(f)['people'][0]['pose_keypoints_2d']

    alphapose_kps = np.array(alphapose_kps).reshape(-1, 3)
    plt.imshow(plt.imread(os.path.join(root_dir, 'frames', frame)))
    for kp_idx, kp in enumerate(given_kps):
        plt.text(kp[0], kp[1], kp_idx, color='r')

    for kp_idx, kp in enumerate(alphapose_kps):
        plt.text(kp[0], kp[1], kp_idx, color='b')

    plt.title(frame)

    plt.show()