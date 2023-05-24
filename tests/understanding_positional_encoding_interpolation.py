import torch
from matplotlib import pyplot as plt

from eg3d.training.volumetric_rendering.mh_projector import PosEncGivenPos

enc_dim = 32
pix_loc_pe = PosEncGivenPos(enc_dim, enc_type='learned')

loc = torch.Tensor([[[0.1, -1, -1.0,]]])
interpolated_feature = pix_loc_pe(loc)[0, 0]

corr_list = []
for i in range(32):
    indexed_feature = pix_loc_pe.pos_enc[0, :, i, 0, 0]
    corr_list.append((indexed_feature * interpolated_feature).sum().item())

expected_pos = ((loc[0, 0, 0] + 1) / 2) * 31
print(expected_pos)
plt.plot(corr_list)
plt.plot(corr_list, 'o')
plt.plot([expected_pos, expected_pos], [min(corr_list), max(corr_list)])
plt.show()