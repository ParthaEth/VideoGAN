import numpy as np
import torch
import matplotlib.pyplot as plt


def make_mask(rows, cols, time_steps, neighbourhood_to_attend):
    seq_len = np.prod([rows, cols, time_steps])
    mask_for_img_pts = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask_row = 0
    for row in range(rows):
        for col in range(cols):
            for time in range(time_steps):
                mask_this_row = torch.ones((rows, cols, time_steps), dtype=torch.bool)
                first_neighbour_row = max(0, row - neighbourhood_to_attend // 2)
                first_neighbour_col = max(0, col - neighbourhood_to_attend // 2)
                first_neighbour_time = max(0, time - neighbourhood_to_attend // 2)

                mask_this_row[first_neighbour_row:row + neighbourhood_to_attend // 2,
                              first_neighbour_col:col + neighbourhood_to_attend // 2,
                              first_neighbour_time:time + neighbourhood_to_attend // 2, ] = False
                mask_for_img_pts[mask_row, :] = mask_this_row.flatten()
                mask_row += 1

    return mask_for_img_pts.squeeze()

# mask_for_img_pts = make_mask(16, 16, 16, 4)
mask_for_img_pts = make_mask(64, 64, 1, 96)
# plt.imshow(mask_for_img_pts[1556, :].reshape((16, 16, 16))[:, :, 6])
plt.imshow(mask_for_img_pts)
plt.show()
