import os
import imageio
import tqdm


root_dir = '/is/cluster/fast/pghosh/datasets/FACEFORENSICS/frames/train/original'
outdir = os.path.join(root_dir, '../ffhq_crp_mp4_256X256X256')
os.makedirs(outdir, exist_ok=True)
original_vids = os.listdir(root_dir)

for vid in tqdm.tqdm(original_vids):
    frames = os.listdir(os.path.join(root_dir, vid))
    if len(frames) < 256:
        continue
    video_out = imageio.get_writer(os.path.join(outdir, f'{vid}.mp4'), mode='I', fps=60, codec='libx264')

    f_c = 0
    for frame in sorted(frames):
        frm_img = imageio.imread(os.path.join(root_dir, vid, frame))
        video_out.append_data(frm_img)
        f_c += 1
        if f_c == 256:
            break

    video_out.close()
