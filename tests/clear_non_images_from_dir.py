from PIL import Image
import tqdm
import os

in_dir = '/is/cluster/fast/pghosh/datasets/celebV_HQ/frames'
expected_size = [256, 256]

for img in tqdm.tqdm(os.listdir(in_dir)):
    img_full_path = os.path.join(in_dir, img)
    try:
        with Image.open(img_full_path) as im:
          if list(im.size) != expected_size:
              print(f'removing {img_full_path}')
              os.remove(img_full_path)
    except:
        os.remove(img_full_path)
        print(f'removing {img_full_path}')