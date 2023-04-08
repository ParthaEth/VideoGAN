import numpy as np
from PIL import Image, ImageDraw
import imageio
import argparse
import tqdm


def main(start_idx, end_idx, disable_progressbar):
    height = width = 256
    total_frames = 256

    size_h = size_w = 80
    outdir = '/is/cluster/fast/pghosh/datasets/bouncing_sq/no_motion'
    if disable_progressbar:
        pbar = range(start_idx, end_idx)
    else:
        pbar = tqdm.tqdm(range(start_idx, end_idx))
    for vid_idx in pbar:
        x0 = np.random.randint(0, 256 - size_w)
        y0 = np.random.randint(0, 256 - size_h)
        vel_p_frame = [0, 0]  # np.random.uniform(0.1, 3, 2)  # velocity in pixels per frame
        color = tuple(np.random.randint(0, 255, 3))

        np.save(f'{outdir}/init_cond/{vid_idx:05d}.npy', np.array([x0, y0, vel_p_frame[0], vel_p_frame[1]]))

        video_out = imageio.get_writer(f'{outdir}/vids/{vid_idx:05d}.mp4', mode='I', fps=60, codec='libx264')
        for time_idx in range(total_frames):
            x_next = x0 + vel_p_frame[0]
            if x_next + size_w > width:
                vel_p_frame[0] *= -1
                x0 -= (x_next + size_w - width)
            elif x_next < 0:
                vel_p_frame[0] *= -1
                x0 -= x_next
            else:
                x0 = x_next

            y_next = y0 + vel_p_frame[1]
            if y_next + size_h > height:
                vel_p_frame[1] *= -1
                y0 -= (y_next + size_h - height)
            elif y_next < 0:
                vel_p_frame[1] *= -1
                y0 -= y_next
            else:
                y0 = y_next

            shape = [int(x0), int(y0), int(x0 + size_w), int(y0 + size_h)]
            image = Image.new('RGB', (height, width))
            img1 = ImageDraw.Draw(image)
            img1.rectangle(shape, fill=color)
            # print((x0, y0))
            # plt.imshow(image)
            # plt.show()
            video_out.append_data(np.array(image))

        if not disable_progressbar:
            pbar.set_description(f'{vid_idx}')

        video_out.close()


if __name__ == '__main__':
    total_processes = 1000
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-r", "--rank", type=int, help="rank of this process")
    argParser.add_argument("-dp", "--disable_progressbar", type=bool, help="don't show progress")
    args = argParser.parse_args()
    rank = args.rank
    st_end_idx = np.linspace(0, 70_000, total_processes).astype(int)
    st_idx = st_end_idx[rank]
    end_idx = st_end_idx[rank + 1]
    main(st_idx, end_idx, args.disable_progressbar)