import os
import imageio
import numpy as np
from PIL import Image

# Define your video directories
video_directories = \
    ['/is/cluster/fast/pghosh/ouputs/video_gan_runs/ten_motions/00086-ffhq-ffhq_X_10_good_motions_10_motions-gpus8-batch32-gamma1/video',
     '/is/cluster/fast/pghosh/ouputs/video_gan_runs/sky_timelapse/00019-ffhq-train_clips-gpus8-batch128-gamma1/videos_160_frames',
     '/is/cluster/fast/pghosh/ouputs/video_gan_runs/fashion_vids/bdmm/00021-ffhq-fasion_video_bdmm-gpus8-batch128-gamma1/video']
video_idxs = {video_directories[0]: [1118], video_directories[1]: [0], video_directories[2]: [123],}

# Set parameters
n_frames = 12
output_width = 256 * n_frames  # Width of the output image
output_height = 256  # Height of the output image
frame_interval = 13  # Interval between frames (e.g., every 30 frames)

# Create an output directory
output_directory = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/paper_images/teaser'
os.makedirs(output_directory, exist_ok=True)


# Function to extract equally spaced frames from a video
def extract_frames(video_path, frame_interval):
    frames = []
    with imageio.get_reader(video_path, 'ffmpeg') as video:
        num_frames = frame_interval * n_frames
        for frame_num in range(0, num_frames, frame_interval):
            frame = video.get_data(frame_num)
            frames.append(frame)
    return frames


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

    # Calculate the width of each frame in the output image
    frame_width = output_width // len(output_frames)

    # Resize and paste frames side by side
    for j, frame in enumerate(output_frames):
        frame = Image.fromarray(frame)
        frame = frame.resize((frame_width, output_height))
        output_image.paste(frame, (j * frame_width, 0))

    # Save the output image
    output_image.save(os.path.join(output_directory, f'{i}.png'))

print(f'Images created and saved in the output directory: \n {output_directory}.')
