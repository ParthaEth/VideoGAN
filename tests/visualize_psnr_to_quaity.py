# import matplotlib.pyplot as plt
# from skimage import data, img_as_float
# from skimage.filters import gaussian
# from skimage.metrics import peak_signal_noise_ratio
#
# # Load an example image (you can replace this with your own image)
# image = img_as_float(data.camera())
#
# # Apply Gaussian blur with different sigma values
# sigma_values = [0.625, 1.2, 2.5]
# blurred_images = [gaussian(image, sigma=sigma) for sigma in sigma_values]
#
# # Calculate PSNR for each blurred image
# psnrs = [peak_signal_noise_ratio(image, blurred) for blurred in blurred_images]
#
# # Display images with captions
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# for i, (sigma, blurred, psnr) in enumerate(zip(sigma_values, blurred_images, psnrs)):
#     ax = axes[i]
#     ax.imshow(blurred, cmap='gray')
#     ax.set_title(f'Sigma: {sigma:.2f}\nPSNR: {psnr:.2f} dB')
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()

import numpy as np

MSE = 0.015484
MAX = 2

PSNR = 20 * np.log10(MAX) - 10 * np.log10(MSE)
print("PSNR:", PSNR, "dB")