# from scipy import datasets
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import imageio


fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
fname = '/home/pghosh/Downloads/seed0004.mp4'

reader = imageio.get_reader(fname, mode='I')
vid_vol = []
for im in reader:
    vid_vol.append(im)
vid_vol = np.stack(vid_vol, axis=0).transpose(1, 2, 3, 0)

ascent = vid_vol[:, :, :, 0]

result = gaussian_filter(ascent, sigma=(10, 10), axes=(0, 1))
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()