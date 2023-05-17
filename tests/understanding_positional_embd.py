import torch
from positional_encodings.torch_encodings import PositionalEncoding1D
import matplotlib.pyplot as plt

p_enc_1d_model = PositionalEncoding1D(10)

# Return the inputs with the position encoding added
# p_enc_1d_model_sum = Summer(PositionalEncoding1D(10))

x = torch.rand(1, 200, 10)
penc_no_sum = p_enc_1d_model(x)
plt.plot(penc_no_sum[0,:, ::2])  #Looking at different position's vale given by this dimension of the embedding vector
plt.show()