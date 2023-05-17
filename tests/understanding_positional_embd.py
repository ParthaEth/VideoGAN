import torch
# from positional_encodings.torch_encodings import PositionalEncoding1D
import matplotlib.pyplot as plt
def spat_encodings_3d(n):
    """

    Returns a 1 x d x n x n x n feature tensor containing sines and cosines of all
    possible frequences
    d = 3n-3
    """
    x_ = torch.arange(n).view(-1, 1, 1).expand(-1, n, n) / n
    y_ = torch.arange(n).view(1, -1, 1).expand(n, -1, n) / n
    z_ = torch.arange(n).view(1, 1, -1).expand(n, n, -1) / n
    spat_enc = torch.stack((x_, y_, z_), dim=0)
    pos_enc = torch.cat([
        torch.cat((torch.sin(k * 2 * torch.pi * spat_enc),
                    torch.cos(k * 2 * torch.pi * spat_enc)), dim=0)
        for k in range(1, n // 2)
    ], dim=0)
    pos_enc = torch.cat((pos_enc, spat_enc), dim=0)
    return pos_enc.unsqueeze(0)


pe = spat_encodings_3d(16)
plt.plot(pe[0, :, 5, 5, 1])
plt.show()