import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import filtered_lrelu
from ..networks_stylegan3 import FullyConnectedLayer, modulated_conv2d



class GroupNorm32(torch.nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 w_dim,                          # Intermediate latent (W) dimensionality.
                 resolution,
                 kernel_size=3,  # Convolution kernel size. Ignored for final the ToRGB layer.
                 up=1,
                 is_torgb = False,                       # Is this the final ToRGB layer?
                 is_critically_sampled = False,          # Does this layer use critical sampling?
                 use_fp16=False,                       # Does this layer use FP16?
                 in_sampling_rate = 0.0,               # Input sampling rate (s).
                 out_sampling_rate = 0.0,              # Output sampling rate (s).
                 in_cutoff = 0.0,                      # Input cutoff frequency (f_c).
                 out_cutoff = 0.0,                     # Output cutoff frequency (f_c).
                 in_half_width = 0.0,                  # Input transition band half-width (f_h).
                 out_half_width = 0.0,                 # Output Transition band half-width (f_h).

                 # Hyperparameters.
                 filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
                 lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
                 use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
                 conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
                 magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
                 residual: bool = False,
                 gn_groups: int = 32,  # Number of groups for GroupNorm

    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.up = up
        self.in_size = np.array([self.resolution // self.up, self.resolution // self.up])
        self.out_size = np.array([self.resolution, self.resolution])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else kernel_size
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta
        if residual: assert in_channels == out_channels
        self.residual = residual

        if self.residual:
            layer_scale_init = 1e-5
            assert up == 1
            self.norm = GroupNorm32(gn_groups, out_channels)
            self.gamma = torch.nn.Parameter(layer_scale_init * torch.ones([1, out_channels, 1, 1])).to(memory_format=torch.contiguous_format)

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, noise_mode='random', force_fp32=False, update_emas=False):
        assert noise_mode in ['random', 'const', 'none'] # unused
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        if self.residual:
            x = self.norm(x)

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        y = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        y = filtered_lrelu.filtered_lrelu(x=y, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        misc.assert_shape(y, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert y.dtype == dtype

        if self.residual:
            y = self.gamma * y
            y = y.to(dtype).add_(x).mul(np.sqrt(2))

        return y
