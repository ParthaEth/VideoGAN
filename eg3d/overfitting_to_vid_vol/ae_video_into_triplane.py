import sys
sys.path.append('../')
import os
import torch
import numpy as np
import random

import imageio
import tqdm
from training.triplane import OSGDecoder
from training.volumetric_rendering.renderer import AxisAligndProjectionRenderer
from overfitting_to_vid_vol.tri_plane_encoder import TriplaneEncoder
from training.dataset import VideoFolderDataset
from torch_utils import misc


def save_video(recon_vids, original_vids, save_dir):
    def save_vid(vid, full_path):
        vid = torch.clip((vid.permute(3, 0, 1, 2) + 1) * 127.5, 0, 255)
        video_out = imageio.get_writer(full_path, mode='I', fps=30, codec='libx264')
        for video_frame in vid:
            video_out.append_data(video_frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        video_out.close()

    os.makedirs(save_dir, exist_ok=True)
    for i, (recon_vid, original_vid) in enumerate(zip(recon_vids, original_vids)):
        save_vid(recon_vid, os.path.join(save_dir, f'recon_vid_{i}.mp4'))
        save_vid(original_vid, os.path.join(save_dir, f'original_vid_{i}.mp4'))


torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

device = 'cuda'
input_vid_res = 3, 256, 256, 256
motion_features = 9
appearance_feat = 9
tri_plane_res = appearance_feat + motion_features, 64, 64
over_fit = True
dataset_dir = '/is/cluster/fast/pghosh/datasets/ffhq_X_celebv_hq_1_motions'
out_dir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting'
restore_from = None  # '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/enc_rend_and_dec_best.pth'
os.makedirs(out_dir, exist_ok=True)
b_size = 4  # 32

encoder = TriplaneEncoder(input_vid_res, tri_plane_res).to(device)
enc_params = [param for param in encoder.parameters()]

renderer = AxisAligndProjectionRenderer(return_video=True, motion_features=motion_features).to(device)
rend_params = [param for param in renderer.parameters()]

decoder = OSGDecoder(appearance_feat, {'decoder_lr_mul': 1, 'decoder_output_dim': 32}).to(device)
dec_params = [param for param in decoder.parameters()]

# resume from checkpoint
if restore_from is not None:
    chk_pt = torch.load(restore_from)
    encoder.load_state_dict(chk_pt['encoder'])
    decoder.load_state_dict(chk_pt['decoder'])
    renderer.load_state_dict(chk_pt['renderer'])
    print(f'restored from {restore_from}')

mdl_params = rend_params + enc_params + dec_params
opt = torch.optim.Adam(mdl_params, lr=1e-3, betas=(0.5, 0.9))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.25, patience=300, verbose=True,
                                                       threshold=1e-3)

train_itr = 5000
criterion = torch.nn.MSELoss()
pbar = tqdm.tqdm(range(train_itr))
losses = []
losses_lr = []
best_psnr = 0

training_set = VideoFolderDataset(path=dataset_dir, return_video=True, cache_dir=None,
                                  fixed_time_frames=True, time_steps=-1, blur_sigma=0, use_labels=True)

training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=0, num_replicas=1, seed=0)
training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                         batch_size=b_size, pin_memory=False,
                                                         worker_init_fn=training_set.worker_init_fn, num_workers=0))

_, peep_vid_real, cond = next(training_set_iterator)
peep_vid_real = peep_vid_real.to(device).to(torch.float32) / 127.5 - 1
for i in pbar:
    if i != 0 and not over_fit:
        _, peep_vid_real, cond = next(training_set_iterator)
        peep_vid_real = peep_vid_real.to(device).to(torch.float32) / 127.5 - 1
    cond = cond.to(device)

    planes_batch = encoder(peep_vid_real)
    colors_coarse, peep_vid, features, flows_and_mask = renderer(planes_batch, decoder, cond, None,
                                                                 {'density_noise': 0, 'box_warp': 0.999,
                                                                  'neural_rendering_resolution': input_vid_res[1]})
    # import ipdb; ipdb.set_trace()
    loss = criterion(peep_vid, peep_vid_real[:, :, ::2, ::2, ::8])
    loss.backward()
    # import ipdb; ipdb.set_trace()
    opt.step()
    opt.zero_grad()
    losses.append(loss.item())
    psnr_lr = 10 * np.log10(4 / np.mean(losses[-10:]))
    if best_psnr < psnr_lr:
        # import ipdb; ipdb.set_trace()
        best_psnr = psnr_lr
        torch.save({'renderer': renderer.state_dict(), 'decoder': decoder.state_dict(),
                    'encoder': encoder.state_dict()},
                   os.path.join(out_dir, f'enc_rend_and_dec_PSNR_{psnr_lr:0.2f}.pth'))

    pbar.set_description(f'loss: {np.mean(losses[-10:]):0.6f} PSNR: {psnr_lr:0.2f}, PSNR_best:{best_psnr:0.2f}')
    scheduler.step(np.mean(losses[-10:]))

    if i % 200 == 199:
        # Save original and reconstructed video
        save_video(peep_vid.detach(), peep_vid_real, os.path.join(out_dir, f'{i}'))

# print(rend_params)
