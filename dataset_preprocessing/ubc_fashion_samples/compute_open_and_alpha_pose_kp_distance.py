import subprocess
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_pose_mahalanobis_dist(open_pose_kps, alphapose_kps):
    alphapose_2_openpose_idx = {2: 2, 1: 1, 5: 5, 3: 3, 4: 4, 8: 9, 11: 12, 7: 7, 0: 0, 16: 17, 14: 15, 15: 16, 17: 18,
                                9: 10, 12: 13, 10: 11, 13: 14, 6: 6}
    pix_mh_dist = 0
    if np.sum(alphapose_kps[:, 2] > 0.5) > 5 and np.sum(open_pose_kps[:, 2] > 0.5) > 5:
        alphapose_kps[alphapose_kps[:, 2] < 0.5, 2] = 0
        open_pose_kps[open_pose_kps[:, 2] < 0.5, 2] = 0
        normalize = 0
        for alpha_kp_idx, alphapose_kp in enumerate(alphapose_kps):
            pix_dist = np.linalg.norm(alphapose_kp[0:2] - open_pose_kps[alphapose_2_openpose_idx[alpha_kp_idx], :2])
            pix_mh_dist += pix_dist * alphapose_kp[2] * open_pose_kps[alphapose_2_openpose_idx[alpha_kp_idx], 2]
            if alphapose_kp[2] * open_pose_kps[alphapose_2_openpose_idx[alpha_kp_idx], 2] != 0:
                normalize += 1
        pix_mh_dist /= normalize
    else:
        pix_mh_dist = 99999

    return pix_mh_dist


def compare_all_frames(path_UBC_fashion_smpl, gen_vid_dir, visualize):
    frames = sorted(os.listdir(gen_vid_dir))
    if len(frames) < 256:  # clip is too small
        return None
    dists = []
    video_name = os.path.basename(gen_vid_dir)
    driving_vid = video_name.split('____')[0]
    root_dir = os.path.dirname(gen_vid_dir)
    kp_dir = f'{root_dir}_kp'
    for frame in frames:
        given_kps_file = frame.replace('.png', '.json')
        with open(os.path.join(path_UBC_fashion_smpl, driving_vid, 'kptsmpls', given_kps_file)) as f:
            open_pose_kps = np.array(json.load(f)['pose_keypoints_2d'])

        with open(os.path.join(kp_dir, video_name, 'sep-json', given_kps_file)) as f:
            alphapose_kps = json.load(f)['people'][0]['pose_keypoints_2d']
        alphapose_kps = np.array(alphapose_kps).reshape(-1, 3)

        dists.append(get_pose_mahalanobis_dist(open_pose_kps, alphapose_kps))
        if visualize and dists[-1] > 80:
            plt.imshow(plt.imread(os.path.join(gen_vid_dir, frame)))
            for kp_idx, kp in enumerate(open_pose_kps):
                plt.text(kp[0], kp[1], kp_idx, color='r')

            for kp_idx, kp in enumerate(alphapose_kps):
                plt.text(kp[0], kp[1], kp_idx, color='b')

            plt.title(f'{frame} dist: {dists[-1]}')
            plt.show()

    return np.array(dists)


def make_video(image_directory, output_video):

    if os.path.exists(output_video):
        return

    # Set the frame rate (frames per second)
    frame_rate = 30  # You can adjust this as needed
    num_frames_to_encode = 256

    # Use subprocess to call the ffmpeg command
    command = [
        "ffmpeg",
        "-framerate", str(frame_rate),
        "-pattern_type", "glob",
        "-i", f"{image_directory}/frame_*.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vframes", str(num_frames_to_encode),  # Encode only the first 256 frames
        output_video
    ]

    subprocess.run(command)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-r", "--rank", type=int, help="rank of this process")
    args = argParser.parse_args()

    vids_per_process = 10

    path_UBC_fashion_smpl = '/is/cluster/fast/scratch/ssanyal/video_gan/fashion_videos/UBC_fashion_smpl/train'
    root_dir = '/is/cluster/fast/scratch/ssanyal/video_gan/fashion_videos/GENERATED_VIDEOS_3/'
    dest_video_dir = '/is/cluster/fast/pghosh/datasets/fasion_video_bdmm/'
    thresshold_dist = 10
    vid_list = sorted(os.listdir(root_dir))
    start_id = args.rank * vids_per_process
    stop_id = start_id + vids_per_process
    for vid in vid_list[start_id:stop_id]:
        try:
            distances = compare_all_frames(path_UBC_fashion_smpl, os.path.join(root_dir, vid), visualize=False)
            if distances is None or np.sum(distances > thresshold_dist) > 20:
                continue
        except FileNotFoundError as e:
            print(e)
            continue
        # The video is legit make a video out of it
        print(max(distances))
        make_video(os.path.join(root_dir, vid), os.path.join(dest_video_dir, vid))
