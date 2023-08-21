import numpy as np
import torch
import tqdm


class TriplaneParams(torch.nn.Module):
    def __init__(self, n_codes):
        super().__init__()
        self.params = torch.nn.Parameter(torch.randn(n_codes, 32, 64, 64, 3, dtype=torch.float32))


if __name__ == '__main__':
    n_codes = 7000
    triplanes = TriplaneParams(n_codes)
    minibatch = 64
    optm = torch.optim.Adam(triplanes.parameters(), lr=10)
    grad = torch.randn(minibatch, 32, 64, 64, 3, dtype=torch.float32)

    for i in tqdm.tqdm(range(1000)):
        optm.zero_grad()
        idx_s = np.random.randint(0, n_codes, minibatch)
        triplanes.params[idx_s].grad = grad
        optm.step()
