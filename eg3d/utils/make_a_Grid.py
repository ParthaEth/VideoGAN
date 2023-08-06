import os
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

dir_name = '/home/pghosh/Downloads/stylegan_xl/my_training/'
grid_h = 6
grid_wdith = 10

file_idx = 0
images = []
for img_file in sorted(os.listdir(dir_name)):
    if img_file.endswith('.png'):
        np_imgs = np.array(Image.open(os.path.join(dir_name, img_file))).astype(np.float32) / 255
        img_tensor = torch.from_numpy(np_imgs.transpose(2, 0, 1))
        images.append(img_tensor)
save_image(images, os.path.join(dir_name, 'combined_grid.png'), nrow=grid_wdith, pad_vallue=255)
