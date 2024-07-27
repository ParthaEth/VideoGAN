import cv2
import numpy as np
import os
import argparse


def get_filename(full_path):
    # Extract the base name from the full path
    base_name = os.path.basename(full_path)
    # Split the base name into the name and extension
    file_name_without_extension, _ = os.path.splitext(base_name)
    return file_name_without_extension


def process_videos(video_files, start_index, end_index, output_dir):
    for idx in range(start_index, min(end_index, len(video_files))):
        video_path = video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        clip_length = 161  # 5 seconds at 30 fps (actually 32 fps)

        # Process in chunks of clip_length frames (non-overlapping)
        for start_frame in range(0, frame_count, clip_length):
            if start_frame + clip_length > frame_count:
                break  # Skip the last clip if it has less than clip_length frames

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(clip_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if len(frames) == clip_length:
                # No 256X256 patch is available that completely avoids the watermark
                # # Ensuring no overlap with the watermark
                # valid_start_y = [range(0, max(194 - 256, 0)), range(246, frame_height - 256)]
                # valid_ranges = [r for r in valid_start_y if r]  # Filter out empty ranges
                # if not valid_ranges:
                #     continue  # Skip if no valid range exists
                #
                # # Choose a random range and then a random y within that range
                # chosen_range = np.random.choice(valid_ranges)
                # y = np.random.choice(chosen_range)
                # x = np.random.randint(0, frame_width - 256)
                y = (frame_height - 256) // 2
                x = (frame_width - 256) // 2

                clip_stem = os.path.join(output_dir, f'{get_filename(video_path)}_clip_{start_frame // clip_length}')
                clip_output_path = f'{clip_stem}.mp4'
                out = None

                try:
                    # Initialize video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'X264'
                    out = cv2.VideoWriter(clip_output_path, fourcc, 30.0, (256, 256))

                    # Apply the same random crop to each frame and write to video file
                    for frame in frames:
                        cropped = frame[y:y + 256, x:x + 256]
                        out.write(cropped)

                finally:
                    if out is not None:
                        out.release()

        cap.release()


def main():
    parser = argparse.ArgumentParser(description='Process a subset of videos to extract frames.')
    parser.add_argument('--run_id', type=int, default=0, help='Unique identifier for the current run')
    parser.add_argument('--total_runs', type=int, default=1, help='Total number of distributed instances')
    parser.add_argument('-od', '--output_directory', type=str, default='samples',
                        help='Directory to save the output clips and frames')
    args = parser.parse_args()

    video_dir = '/is/cluster/scratch/pghosh/dataset/WebVid_10M/flower_train_vids'  # Update with the path to your videos
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]

    videos_per_run = len(video_files) // args.total_runs
    start_index = args.run_id * videos_per_run
    end_index = start_index + videos_per_run if args.run_id < args.total_runs - 1 else len(video_files)

    os.makedirs(args.output_directory, exist_ok=True)
    print(f"Processing videos {start_index} to {end_index} out of {len(video_files)}")
    print(f"Output directory: {args.output_directory}")

    try:
        process_videos(video_files, start_index, end_index, args.output_directory)
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
