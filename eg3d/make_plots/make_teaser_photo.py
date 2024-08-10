import os
import imageio
import numpy as np
from PIL import Image

###################################### STYLE_GANT_OURS ###############################################################
# # Define your video directories This si for styleGAN_T
# video_directories = \
#     ['/is/cluster/fast/pghosh/ouputs/video_gan_runs/ten_motions/00086-ffhq-ffhq_X_10_good_motions_10_motions-gpus8-batch32-gamma1/video',
#      '/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00021-ffhq-fasion_video_bdmm-gpus8-batch128-gamma1/video',
#      '/is/cluster/fast/pghosh/ouputs/video_gan_runs/webvid_10M_flowers/stg_t/random_crops/good_motion']
#
#
#
# # for x in range(10, 15):
#
# # video_idxs = {video_directories[0]: [7],  video_directories[1]: [123], video_directories[2]: [7]}
# # video_idxs = {video_directories[0]: [1118],  video_directories[1]: [123], video_directories[2]: [9]}
# # video_idxs = {video_directories[0]: [155], video_directories[1]: [150], video_directories[2]: [7]}
#
# video_idxs = {video_directories[0]: [159], video_directories[1]: [151], video_directories[2]: [1]}



# # Create an output directory
# # output_directory = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/paper_images/teaser/flowers'
# output_directory = \
#     '/is/cluster/fast/pghosh/ouputs/video_gan_runs/webvid_10M_flowers/stg_t/random_crops/good_motion/teaser/'

########################################### STYLEGAN_T ends ###############################################################

########################################### MOCO_GAN ###############################################################
video_directories = \
    ['/is/cluster/scratch/ssanyal/video_gan/mocogan/fashion_video_bdmm_all/output/network-snapshot-002073_FID_trunc_0.7',
     '/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00021-ffhq-fasion_video_bdmm-gpus8-batch128-gamma1/video',
     '/is/cluster/scratch/ssanyal/video_gan/stylegan_V/mocogan/webvid10M_flowers/output/network-snapshot-022809_fps150']

video_idxs = {video_directories[0]: [7],  video_directories[1]: [123], video_directories[2]: [7]}
output_directory = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/paper_images/teaser/moco_gan/'
########################################### MOCO_GAN ends ###############################################################


# ########################################### STYLEGAN_V ###############################################################
# video_directories = \
#     ['/is/cluster/scratch/ssanyal/video_gan/stylegan_V/ffhq_10_motions/output_ffhq_10motions/out_videos_14515',
#      '/is/cluster/scratch/ssanyal/video_gan/stylegan_V/fasion_video_bdmm_all/more_frames/output/vids_019699',
#      '/is/cluster/scratch/ssanyal/video_gan/stylegan_V/web10mflowers/output/network-snapshot-025000_fps150']
#
# video_idxs = {video_directories[0]: [9],  video_directories[1]: [3], video_directories[2]: [150]}
# output_directory = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/paper_images/teaser/styleGAN_V/'
########################################### STYLEGAN_V ends ###############################################################



os.makedirs(output_directory, exist_ok=True)
# Set parameters
overlay = True
if overlay:
    n_frames = 6
    output_width = 256 * n_frames // 2
    frame_width = output_width // n_frames * 2
else:
    n_frames = 3
    output_width = 256 * n_frames  # Width of the output image
    # Calculate the width of each frame in the output image
    frame_width = output_width // n_frames

output_height = 256  # Height of the output image
# frame_interval = 13  # Interval between frames (e.g., every 30 frames)
frame_interval = 21  # Interval between frames (e.g., every 30 frames)


# Function to extract equally spaced frames from a video
def extract_frames(video_path, frame_interval):
    frames = []
    with imageio.get_reader(video_path, 'ffmpeg') as video:
        num_frames = frame_interval * n_frames
        for frame_num in range(0, num_frames, frame_interval):
            frame = video.get_data(frame_num)
            frames.append(frame)
    return frames


def overlay_2frames(frames):
    frame1 = frames[0]
    frame2 = frames[1]
    frame1 = Image.fromarray(frame1)
    frame2 = Image.fromarray(frame2)
    frame1_r, frame1_g, frame1_b = frame1.split()
    frame2_r, frame2_g, frame2_b = frame2.split()
    # return Image.merge('RGB', (frame1_r, frame1_g, frame2_b))
    return Image.merge('RGB', (frame2_r, frame1_g, frame1_b))


def get_r_ch_overlay(image_list, output_image, frame_width):
    overlayed_frames = []
    for i in range(0, len(image_list), 2):
        overlayed_frames.append(overlay_2frames(image_list[i:i+2]))

    for i, frame in enumerate(overlayed_frames):
        frame = frame.resize((frame_width, output_height))
        output_image.paste(frame, (i * frame_width, 0))

    return output_image


# Loop through the video directories
for i, video_directory in enumerate(video_directories):
    output_frames = []
    for video_file in np.array(sorted(os.listdir(video_directory)))[video_idxs[video_directory]]:
        if video_file.endswith('.mp4'):  # Change the video format if needed
            video_path = os.path.join(video_directory, video_file)
            frames = extract_frames(video_path, frame_interval)
            output_frames.extend(frames)

    # Create a blank canvas for the output image
    output_image = Image.new('RGB', (output_width, output_height))



    # Resize and paste frames side by side
    if overlay:
        output_image = get_r_ch_overlay(output_frames, output_image, frame_width)
    else:
        for j, frame in enumerate(output_frames):
            frame = Image.fromarray(frame)
            frame = frame.resize((frame_width, output_height))
            output_image.paste(frame, (j * frame_width, 0))

    # Save the output image
    output_image.save(os.path.join(output_directory, f'{video_idxs[video_directories[i]][0]}_{video_file[:-4]}.png'))

print(f'Images created and saved in the output directory: \n {output_directory}.')
