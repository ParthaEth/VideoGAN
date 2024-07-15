import sys
sys.path.append('../')
import torch
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

# network_pkl = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/ucf101/00026-ffhq-clips-gpus4-batch256-gamma1/network-snapshot-000000.pkl'
# backbone = 'StyleGANXL'

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
synthesis_kwargs = {}
G.rendering_kwargs['neural_rendering_resolution'] = G.neural_rendering_resolution
G.rendering_kwargs['use_cached'] = True
G.rendering_kwargs['dont_print_warning'] = True
with torch.no_grad():
    with torch.autograd.profiler.profile(use_cuda=True, with_flops=True) as prof:
        # flops, params = profile(G, inputs=(z, conditioning_params))
        z = torch.randn(1, G.z_dim, device=device, dtype=torch.float32)
        conditioning_params = torch.zeros((1, G.c_dim), dtype=z.dtype, device=device)
        conditioning_params[:, 2] = 1
        # import ipdb; ipdb.set_trace()
        flops = 0
        ws = G.mapping(z, conditioning_params, truncation_psi=1, truncation_cutoff=None, update_emas=False)
        planes = G.backbone.synthesis(ws, update_emas=False, **synthesis_kwargs)
        planes = planes.view(1, G.num_planes, -1, planes.shape[-2], planes.shape[-1])
        H = W = G.neural_rendering_resolution
        for t_cond in tqdm.tqdm(time_cod):
            conditioning_params[:, 3] = t_cond
            rgb_image, peep_video, features, flows_and_masks = G.renderer(planes, G.decoder, conditioning_params, None,
                                                                          G.rendering_kwargs)
            assert(peep_video is None)
            rgb_image = rgb_image.permute(0, 2, 1).reshape(1, rgb_image.shape[-1], H, W).contiguous()
            feature_image = features.permute(0, 2, 1).reshape(1, features.shape[-1], H, W).contiguous()
            sr_xy_image = G.superresolution(
                rgb_image, feature_image, ws,
                noise_mode=G.rendering_kwargs['superresolution_noise_mode'],
                **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

# Sum up FLOPs from profiler output
total_flops = 0
for event in prof.function_events:
        total_flops += event.flops

print(f"Total FLOPs: {total_flops/1e12}")
