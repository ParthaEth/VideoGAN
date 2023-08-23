# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, peep_vid_real, gain, cur_nimg):
        # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0,
                 pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
                 r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64,
                 neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0,
                 gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                 apply_crop=False):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.apply_crop = apply_crop
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_img = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution,
                                                 update_emas=update_emas)
        return gen_img, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.apply_crop:
            b_size, ch, img_w, img_h = img['image'].shape
            b_size, ch, vid_w, vid_h = img['peep_vid'].shape
            img['image'][:, :, :, :35] = img['image'][:, :, :, :35] * 0 + 1  # make left border white
            img['image'][:, :, :, 220:] = img['image'][:, :, :, 220:] * 0 + 1  # make right border white

            img['peep_vid'][:, :, :, :int(35 * vid_w/img_w), :] = img['peep_vid'][:, :, :, :int(35 * vid_w/img_w), :] * 0 + 1 # make left border white
            img['peep_vid'][:, :, :, int(220 * vid_w/img_w):, :] = img['peep_vid'][:, :, :, int(220 * vid_w/img_w):, :] * 0 + 1  # make right border white

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(
                torch.cat([img['image'],
                torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear',
                                                antialias=True)],dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(
                augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear',
                antialias=True)

            # run augmentation on the video
            b, chn, h, w, d = img['peep_vid'].shape
            vid_as_b_chn_h_wxd = img['peep_vid'].reshape(b, chn, h, w*d)
            # import ipdb; ipdb.set_trace()
            aug_vid = self.augment_pipe(vid_as_b_chn_h_wxd)
            img['peep_vid'] = aug_vid.reshape(b, chn, h, w, d)

        logits, video_logits = self.D(img, c, update_emas=update_emas)
        return logits, video_logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, peep_vid_real, gain, cur_nimg):
        img_logit_to_video_logit_ratio = np.array([1.0, 1.0], dtype=np.float32)
        img_logit_to_video_logit_ratio /= np.linalg.norm(img_logit_to_video_logit_ratio)
        w_i_logit, w_v_logit = img_logit_to_video_logit_ratio
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter,
                                         filter_mode=self.filter_mode)

        # real video resizing
        batch_s, c_ch, v_h, v_w, v_t = peep_vid_real.shape
        peep_vid_real = peep_vid_real.permute(0, 4, 1, 2, 3).resize(batch_s * v_t, c_ch, v_h, v_w)
        peep_vid_real = filtered_resizing(peep_vid_real, size=neural_rendering_resolution//2, f=self.resample_filter,
                                          filter_mode='antialiased')
        peep_vid_real = peep_vid_real\
            .resize(batch_s, v_t, c_ch, neural_rendering_resolution//2, neural_rendering_resolution//2)\
            .permute(0, 2, 3, 4, 1)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            if hasattr(self.G.backbone.generator, 'head_layer_names'):  # blocking grad computation for double safety!
                self.G.backbone.mapping.requires_grad_(False)
                for name in self.G.backbone.synthesis.layer_names:
                    getattr(self.G.backbone.synthesis, name).requires_grad_(
                        name in self.G.backbone.generator.head_layer_names)

            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob,
                                                            neural_rendering_resolution=neural_rendering_resolution)
                gen_logits, gen_video_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/scores/fake_vid', gen_video_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/signs/fake_vid', gen_video_logits.sign())
                loss_Gmain = w_i_logit * torch.nn.functional.softplus(-gen_logits) + \
                             w_v_logit * torch.nn.functional.softplus(-gen_video_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

            # import ipdb;ipdb.set_trace()
            if hasattr(self.G.backbone.generator, 'head_layer_names'): # StyleGANXL
                g_fixed_norm_rec = False
                for name in self.G.backbone.synthesis.layer_names:
                    if name not in self.G.backbone.generator.head_layer_names and not g_fixed_norm_rec:  # not in head layer => fixed params
                        g_fixed_param = getattr(self.G.backbone.synthesis, name).parameters()
                        training_stats.report(f'G/params/fixed_layer/{name}', next(g_fixed_param).norm(2))
                        g_fixed_norm_rec = True
                    if name == self.G.backbone.generator.head_layer_names[0]:  # in head layer => trainable
                        g_var_param = getattr(self.G.backbone.synthesis, name).parameters()
                        training_stats.report(f'G/params/trainable_layer/{name}', next(g_var_param).norm(2))
                        if g_fixed_norm_rec:
                            break
            elif hasattr(self.G.backbone.generator, 'train_mode'):  # StyleGAN-T
                g_fixed_norm_rec = False
                g_trainable_norm_rec = False
                for param_name, params in self.G.backbone.generator.named_parameters():
                    for trainable_layer_name in self.G.backbone.generator.trainable_layers:
                        if param_name.find(trainable_layer_name) > 0 or self.G.backbone.generator.train_mode == 'all':
                            training_stats.report(f'G/params/trainable_layer/{param_name}', params.norm(2))
                            g_fixed_norm_rec = True
                            g_trainable_norm_rec = (self.G.backbone.generator.train_mode == 'all')  # because there is no fixed layers
                        else:
                            training_stats.report(f'G/params/fixed_layer/{param_name}', params.norm(2))
                            g_trainable_norm_rec = True
                    if g_trainable_norm_rec and g_fixed_norm_rec:
                        break

            # import ipdb; ipdb.set_trace()

            d_fix_norm_rec = False
            d_trn_norm_rec = False
            for name, d_params in self.D.named_parameters():
                if name.find('feature_networks') > 0 or name.find('dino') > 0:  # part of feature network
                    if not d_fix_norm_rec:  # report if not recorded yet
                        d_fix_norm_rec = True
                        training_stats.report(f'D/params/fixed_layer/{name}', d_params.norm(2))
                else:
                    if not d_trn_norm_rec:  # report if not recorded yet
                        d_trn_norm_rec = True
                        training_stats.report(f'D/params/trainable_layer/{name}', d_params.norm(2))

                if d_fix_norm_rec and d_trn_norm_rec:    # if both recorded stop iterating
                    break

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob,
                                                            neural_rendering_resolution=neural_rendering_resolution,
                                                            update_emas=True)
                gen_logits, gen_video_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/scores/fake_vid', gen_video_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/signs/fake_vid', gen_video_logits.sign())
                loss_Dgen = w_i_logit * torch.nn.functional.softplus(gen_logits) +\
                            w_v_logit * torch.nn.functional.softplus(gen_video_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                peep_vid_real_temp = peep_vid_real.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw,
                                'peep_vid': peep_vid_real_temp}

                real_logits, real_video_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/scores/real_vid', real_video_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/signs/real_vid', real_video_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = w_i_logit * torch.nn.functional.softplus(-real_logits) + \
                                 w_v_logit * torch.nn.functional.softplus(-real_video_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(
                                outputs=[real_logits.sum() + real_video_logits.sum()],
                                inputs=[real_img_tmp['image'], real_img_tmp['image_raw'], real_img_tmp['peep_vid']],
                                create_graph=True, only_inputs=True, allow_unused=True)

                            r1_grads_image_pen = r1_grads_image_raw_pen = r1_grads_peep_vid_pen = 0
                            if r1_grads[0] is not None and False:  # Todo(Partha): Clean up, switching off r1 penalty on image discrim
                                r1_grads_image_pen = torch.nan_to_num(r1_grads[0]).square().sum([1, 2, 3])
                            if r1_grads[1] is not None and False:  # Todo(Partha): Clean up, switching off r1 penalty on image discrim
                                r1_grads_image_raw_pen = torch.nan_to_num(r1_grads[1]).square().sum([1, 2, 3])
                            if r1_grads[2] is not None:
                                r1_grads_peep_vid_pen = torch.nan_to_num(r1_grads[2]).square().sum([1, 2, 3, 4])
                        r1_penalty = r1_grads_image_pen + r1_grads_image_raw_pen + r1_grads_peep_vid_pen
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']],
                                                           create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
