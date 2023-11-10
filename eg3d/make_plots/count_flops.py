import sys
sys.path.append('../')
import torch
from fvcore.nn import FlopCountAnalysis
import dnnlib
import legacy
import copy
from training.triplane import TriPlaneGenerator
from torch_utils import misc
import tqdm

reload_modules = True
config = 'ffhq'
network_pkl = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00006-ffhq--gpus8-batch128-gamma1/network-snapshot-000327.pkl'
backbone = 'StyleGANT'

# network_pkl = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00014-ffhq-clips-gpus4-batch128-gamma1/network-snapshot-000573.pkl'
# backbone = 'StyleGAN2'
device = torch.device('cuda')
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

if reload_modules:
    print("Reloading Modules!")

    init_kwargs = copy.deepcopy(G.init_kwargs)
    init_kwargs['use_flow'] = True
    init_kwargs['backbone'] = backbone
    init_kwargs.rendering_kwargs.update({'render_peep_vid': False})
    if config.lower() == 'ffhq' or config.lower() == 'fashion_video':
        init_kwargs.rendering_kwargs.update({'global_flow_div': 16, 'local_flow_div': 64})
    elif config.lower() == 'sky_timelapse':
        init_kwargs.rendering_kwargs.update({'global_flow_div': 16, 'local_flow_div': 16})
    else:
        raise ValueError(f'configuration {config} unknown')

    G_new = TriPlaneGenerator(*G.init_args, **init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G = G_new

time_cod = torch.linspace(0, 1, 160).to(device)
with torch.no_grad():
    # flops, params = profile(G, inputs=(z, conditioning_params))
    z = torch.randn(1, G.z_dim, device=device, dtype=torch.float32)
    conditioning_params = torch.zeros((1, G.c_dim), dtype=z.dtype, device=device)
    conditioning_params[:, 2] = 1
    # import ipdb; ipdb.set_trace()
    flops = 0
    for t_cond in tqdm.tqdm(time_cod):
        conditioning_params[:, 3] = t_cond
        flop_current_pass = FlopCountAnalysis(G, (z, conditioning_params, 1, None, None, False, True, True,)).total()
        flops += flop_current_pass
        print(flop_current_pass / 1e12)
    print(f"Total tera-FLOPs: {flops / 1e12}")
