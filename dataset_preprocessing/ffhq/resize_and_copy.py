from PIL import Image
import os
import tqdm


src_dir = '/is/cluster/pghosh/data/celebA-HQ_images'
dest_dir = '/is/cluster/fast/pghosh/datasets/celebA-HQ_images/'

os.makedirs(dest_dir, exist_ok=True)

max_count = 100
curr_count = 0

for curr_file in tqdm.tqdm(os.listdir(src_dir)):
    if curr_file.endswith('.jpg') or curr_file.endswith('.png'):
        curr_img = Image.open(os.path.join(src_dir, curr_file))
        curr_img = curr_img.resize((512, 512))
        curr_img.save(os.path.join(dest_dir, curr_file))
        curr_count += 1

    if curr_count >= max_count:
        break
