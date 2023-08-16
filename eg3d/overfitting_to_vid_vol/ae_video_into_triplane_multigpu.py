import sys
sys.path.append('../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
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
import dnnlib
import json
import tempfile
from torch_utils import training_stats
from torch_utils import custom_ops


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

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.out_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    # if rank != 0:
    #     custom_ops.verbosity = 'none'

    # Execute training loop.
    t_loop(rank=rank, **c)


def launch_training(c):
    dnnlib.util.Logger(should_flush=True)

    # Create output directory.
    # print('Creating output directory...')
    # os.makedirs(c.run_dir)
    # with open(os.path.join(outdir, 'training_options.json'), 'wt') as f:
    #     json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

        



############################################################################################################


if __name__ == '__main__':
    c = dnnlib.EasyDict()
    
    # device = 'cuda'
    c.input_vid_res = 3, 256, 256, 256
    c.motion_features = 9
    c.appearance_feat = 9
    c.tri_plane_res = c.appearance_feat + c.motion_features, 64, 64
    c.over_fit = False  # True
    c.dataset_dir = '/is/cluster/scratch/ssanyal/project_4/video_data/ffhq_X_celebv_hq_1_motions'
    c.out_dir = '/is/cluster/fast/scratch/ssanyal/video_gan/runs/single_vid_over_fitting'
    c.restore_from = None  # '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/enc_rend_and_dec_best.pth'
    os.makedirs(c.out_dir, exist_ok=True)
    c.b_size = 32
    c.num_gpus = 8
    c.random_seed = 0
    c.train_itr = 5000
    # c.pbar = tqdm.tqdm(range(train_itr))
    c.criterion = torch.nn.MSELoss()

    random.seed(c.random_seed)

    launch_training(c)




#############################################################################################################

def t_loop(rank, random_seed, num_gpus, b_size, dataset_dir, out_dir, input_vid_res, tri_plane_res, motion_features, appearance_feat,
           train_itr, criterion, over_fit, restore_from):

    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = True
    
    #################### Dataset creation ###########################

    training_set = VideoFolderDataset(path=dataset_dir, return_video=True, cache_dir=None,
                                    fixed_time_frames=True, time_steps=-1, blur_sigma=0, use_labels=True)

    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=0)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                            batch_size=b_size, pin_memory=False,
                                                            worker_init_fn=training_set.worker_init_fn, num_workers=8))
    

    ##################### Model creation #######################

    encoder = TriplaneEncoder(input_vid_res, tri_plane_res).to(device)
    enc_params = [param for param in encoder.parameters()]

    renderer = AxisAligndProjectionRenderer(return_video=True, motion_features=motion_features).to(device)
    rend_params = [param for param in renderer.parameters()]

    decoder = OSGDecoder(appearance_feat, {'decoder_lr_mul': 1, 'decoder_output_dim': 32}).to(device)
    dec_params = [param for param in decoder.parameters()]

    print('unitil here ..........................')
    # resume from checkpoint
    if rank==0:
        i = 0
        if restore_from is not None:
            chk_pt = torch.load(restore_from)
            encoder.load_state_dict(chk_pt['encoder'])
            decoder.load_state_dict(chk_pt['decoder'])
            renderer.load_state_dict(chk_pt['renderer'])
            print(f'restored from {restore_from}')

    mdl_params = rend_params + enc_params + dec_params
    
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')

    for param in mdl_params:
        torch.distributed.broadcast(param, src=0)

    # Optimizer creation
    opt = torch.optim.Adam(mdl_params, lr=1e-3, betas=(0.5, 0.9))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.25, patience=300, verbose=True,
    #                                                     threshold=1e-3)


    ############### Training loop #####################

    if rank==0:
        losses = []
        losses_lr = []
        best_psnr = 0


    # _, peep_vid_real, cond = next(training_set_iterator)
    # peep_vid_real = peep_vid_real.to(device).to(torch.float32) / 127.5 - 1
    ## for i in tqdm.tqdm(train_itr):
    while True:
        # if i != 0 and not over_fit:
        #     _, peep_vid_real, cond = next(training_set_iterator)
        #     peep_vid_real = peep_vid_real.to(device).to(torch.float32) / 127.5 - 1
        # cond = cond.to(device)

        _, peep_vid_real, cond = next(training_set_iterator)
        peep_vid_real = peep_vid_real.to(device).to(torch.float32) / 127.5 - 1
        cond = cond.to(device)

        # peep_vid_real = torch.rand([b_size, ] + list(input_vid_res)).to(device)
        # cond = torch.zeros(b_size, 4).to(device)

        planes_batch = encoder(peep_vid_real)
        print('unitil encoder forward pass ..........................')
        colors_coarse, peep_vid, features, flows_and_mask = renderer(planes_batch, decoder, cond, None,
                                                                    {'density_noise': 0, 'box_warp': 0.999,
                                                                    'neural_rendering_resolution': input_vid_res[1]})
        print('unitil decoder forward pass ..........................')
        loss = criterion(peep_vid, peep_vid_real[:, :, ::2, ::2, ::8])
        loss.backward()
        opt.step()
        opt.zero_grad()

        ##################### Save statistics ##############################

        if rank==0:
            i =+ 1
            losses.append(loss.item())
            psnr_lr = 10 * np.log10(4 / np.mean(losses[-10:]))
            if best_psnr < psnr_lr:
                best_psnr = psnr_lr
                torch.save({'renderer': renderer.state_dict(), 'decoder': decoder.state_dict(),
                            'encoder': encoder.state_dict()},
                        os.path.join(out_dir, f'enc_rend_and_dec_PSNR_{psnr_lr:0.2f}.pth'))

            # pbar.set_description(f'loss: {np.mean(losses[-10:]):0.6f} PSNR: {psnr_lr:0.2f}, PSNR_best:{best_psnr:0.2f}')
            print(f'iteration: {i} loss: {np.mean(losses[-10:]):0.6f} PSNR: {psnr_lr:0.2f}, PSNR_best:{best_psnr:0.2f}')
            # scheduler.step(np.mean(losses[-10:]))

            if i % 200 == 199:
                # Save original and reconstructed video
                save_video(peep_vid.detach(), peep_vid_real, os.path.join(out_dir, f'{i}'))

    # print(rend_params)


