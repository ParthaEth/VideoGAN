import os

import ipdb
import numpy as np
import argparse
import scipy.ndimage
import PIL.Image
from PIL import ImageDraw, ImageFilter, ImageChops
from matplotlib import pyplot as plt
import face_alignment

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd.grad_mode import enable_grad
from collections import OrderedDict
import cv2


def image_align_68(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    lm_chin = lm[0: 17, :2]  # left-right
    lm_eyebrow_left = lm[17: 22, :2]  # left-right
    lm_eyebrow_right = lm[22: 27, :2]  # left-right
    lm_nose = lm[27: 31, :2]  # top-down
    lm_nostrils = lm[31: 36, :2]  # top-down
    lm_eye_left = lm[36: 42, :2]  # left-clockwise
    lm_eye_right = lm[42: 48, :2]  # left-clockwise
    lm_mouth_outer = lm[48: 60, :2]  # left-clockwise
    lm_mouth_inner = lm[60: 68, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)
    orig_imag_size = img.size

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # # Crop.
    # border = max(int(np.rint(qsize * 0.1)), 3)
    # crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
    #         int(np.ceil(max(quad[:, 1]))))
    # crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
    #         min(crop[3] + border, img.size[1]))
    # if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
    #     # img = img.crop(crop)
    #     # quad -= crop[0:2]
    #
    # # Pad.
    # pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
    #        int(np.ceil(max(quad[:, 1]))))
    # pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - (crop[2] - crop[0]) + border, 0),
    #        max(pad[3] - (crop[3] - crop[1]) + border, 0))
    # if enable_padding and max(pad) > border - 4:
    #     pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
    #     # img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
    #     new_crop = [crop[0] - pad[0], crop[1] - pad[2], crop[2] + pad[1], crop[3] + pad[3]]
    #     img = img.crop(new_crop)
    #     h, w, _ = img.shape
    #     y, x, _ = np.ogrid[:h, :w, :1]
    #     mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
    #                       1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
    #     blur = qsize * 0.02
    #     img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
    #     img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
    #     img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
    #     quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    dst_file_img = dst_file.split('/')
    temp = [dst_file_img[-1],]
    dst_file_img[-1] = 'aligned_images'
    dst_img_file_cmps = ['/',] + dst_file_img + temp
    dst_img_file = os.path.join(*dst_img_file_cmps)
    img.save(dst_img_file, 'PNG')

    # save transformation settings
    dst_img_file_cmps[-2] = 'alignment_attribs'
    atr_file = os.path.join(*dst_img_file_cmps)
    atr_file = list(atr_file)
    atr_file[-3:] = 'npz'
    atr_file = ''.join(atr_file)
    np.savez(atr_file, quad=quad, shrink=shrink, orig_imag_size=orig_imag_size, transform_size=transform_size)


def reverse_transform(src_file, dst_file, source_overlay_file, original_transform_file):
    # import ipdb; ipdb.set_trace()
    transforms = np.load(original_transform_file)
    img = PIL.Image.open(src_file)

    # import ipdb; ipdb.set_trace()
    # # resize
    # if img.size[0] != transforms['transform_size'][0]:
    #     img = img.resize((transforms['transform_size'][0], transforms['transform_size'][0]), PIL.Image.ANTIALIAS)


    if source_overlay_file is None:
        dst_img = PIL.Image.new('RGB', tuple(transforms['orig_imag_size']))
    else:
        dst_img = PIL.Image.open(source_overlay_file)

    quad = transforms['quad']
    if transforms['shrink'].item() > 1:
        quad = quad * transforms['shrink'].item()

    img = img.resize((int(np.linalg.norm(quad[0] - quad[1])), int(np.linalg.norm(quad[1] - quad[2]))),
                     PIL.Image.ANTIALIAS)
    # mask = PIL.Image.new('L', img.size, color=255)
    # mask = PIL.Image.new('L', img.size, color=255)
    mask = PIL.Image.fromarray((gaussuian_filter(img.size[0], 0.8) * 255/1.0).astype(np.uint8))

    # import ipdb; ipdb.set_trace()

    # Rotate
    theta = np.arctan(-(quad[2][1] - quad[1][1]) / (quad[2][0] - quad[1][0])) * 180 / np.pi
    im_rot = img.rotate(theta, expand=True)
    mask = mask.rotate(theta, expand=True)

    # mask_blur = mask.filter(ImageFilter.GaussianBlur(radius=50))
    # mask2 = ImageChops.multiply(mask, mask_blur)
    # import ipdb;
    # ipdb.set_trace()

    prev_centre = np.mean(quad, axis=0)
    left_t_c = prev_centre - np.array(im_rot.size)/2
    try:
        dst_img.paste(im_rot, tuple(left_t_c.astype(int)), mask=mask)
        # dst_img.paste(im_rot, tuple(left_t_c.astype(int)))

        dst_img.save(dst_file, 'PNG')
    except Exception as e:
        print(f'{dst_file} not generated')


def image_align_24(src_file, dst_file, face_landmarks, output_size=256, transform_size=1024, enable_padding=True):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    lm_chin = lm[0: 3, :2]  # left-right
    lm_eyebrow_left = lm[3: 6, :2]  # left-right
    lm_eyebrow_right = lm[6: 9, :2]  # left-right
    lm_nose = lm[9: 10, :2]  # top-down
    lm_eye_left = lm[10: 15, :2]  # left-clockwise
    lm_eye_right = lm[15: 20, :2]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm[20, :2]
    mouth_right = lm[22, :2]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    # src_file : <PIL>
    img = src_file

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'png')
# Cacaded Face Alignment
class CFA(nn.Module):
    def __init__(self, output_channel_num, checkpoint_name=None):
        super(CFA, self).__init__()

        self.output_channel_num = output_channel_num
        self.stage_channel_num = 128
        self.stage_num = 2

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),

            # nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

        self.CFM_features = nn.Sequential(
            # nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, self.stage_channel_num, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # cascaded regression
        stages = [self.make_stage(self.stage_channel_num)]
        for _ in range(1, self.stage_num):
            stages.append(self.make_stage(self.stage_channel_num + self.output_channel_num))
        self.stages = nn.ModuleList(stages)

        # initialize weights
        if checkpoint_name:
            snapshot = torch.load(checkpoint_name)
            self.load_state_dict(snapshot['state_dict'])
        else:
            self.load_weight_from_dict()

    def forward(self, x):
        feature = self.features(x)
        feature = self.CFM_features(feature)
        heatmaps = [self.stages[0](feature)]
        for i in range(1, self.stage_num):
            heatmaps.append(self.stages[i](torch.cat([feature, heatmaps[i - 1]], 1)))
        return heatmaps

    def make_stage(self, nChannels_in):
        layers = []
        layers.append(nn.Conv2d(nChannels_in, self.stage_channel_num, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(4):
            layers.append(nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(self.stage_channel_num, self.output_channel_num, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def load_weight_from_dict(self):
        model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        weight_state_dict = model_zoo.load_url(model_urls)
        all_parameter = self.state_dict()
        all_weights = []
        for key, value in all_parameter.items():
            if key in weight_state_dict:
                all_weights.append((key, weight_state_dict[key]))
            else:
                all_weights.append((key, value))
        all_weights = OrderedDict(all_weights)
        self.load_state_dict(all_weights)


def gaussuian_filter(kernel_size, sigma=1, muu=0):
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)

    # lower normal part of gaussian
    # normal = 1 / (2, 0 * np.pi * sigma ** 2)

    # Calculating Gaussian filter
    # import ipdb; ipdb.set_trace()
    gauss = np.exp(-((dst - muu) ** 8 / (2.0 * sigma ** 2)))
    return gauss


def get_my_file_list(pid, tot_ps, src_dir):
    """
    pid: int starts from 0 goes till tot_ps - 1
    tot_ps: total processes
    """
    all_files = sorted(os.listdir(src_dir))
    idx_intervals = np.linspace(0, len(all_files), tot_ps + 1).astype(int)
    return all_files[idx_intervals[pid]:idx_intervals[pid + 1]]


if __name__ == '__main__':
    import tqdm
    parser = argparse.ArgumentParser(
        description='A simple script to extract eye and mouth coordinates from a face image.')
    parser.add_argument('-s', '--src', default='/home/pghosh/repos/FaceSwap/Global_Flow_Local_Attention/dataset'
                                               '/FaceForensics/original_sequences/actors/raw/frames'
                                               '/07__talking_against_wall',
                        help='directory of raw images')
    parser.add_argument('-d', '--dst', default='/home/pghosh/repos/FaceSwap/Global_Flow_Local_Attention/dataset/'
                                               'FaceForensics/val_data/07__talking_against_wall',
                        help='directory of aligned images')
    parser.add_argument('-od', '--overlay_dir', default='/home/pghosh/repos/FaceSwap/Global_Flow_Local_Attention/'
                                                        'dataset/FaceForensics/original_sequences/actors/raw/frames/'
                                                        '07__talking_against_wall',
                        help='The directory that contains the iamges over which to overlay de-aligned images')
    parser.add_argument('-o', '--output_size', default=256, type=int, help='size of aligned output (default: 256)')
    parser.add_argument('-t', '--transform_size', default=1024, type=int,
                        help='size of aligned transform (default: 256)')
    parser.add_argument('-a', '--align', default='False', type=lambda x: (str(x).lower() == 'true'),
                        help='Align or dealign')
    parser.add_argument('--no_padding', action='store_false', help='no padding')
    parser.add_argument('-pid', '--process_id', type=int, help='Process id of this job')
    parser.add_argument('-tp', '--total_processes', type=int, help='These many processes are processing at the same '
                                                                   'time on the src dir')

    args = parser.parse_args()
    # import ipdb; ipdb.set_trace()

    if not os.path.exists(args.dst):
        os.mkdir(args.dst)

    if args.align:
        for img_name in tqdm.tqdm(get_my_file_list(args.process_id, args.total_processes, args.src)):
            raw_img_path = os.path.join(args.src, img_name)
            # import ipdb; ipdb.set_trace()
            landmarks_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
            os.makedirs(os.path.join(args.dst, 'aligned_images'), exist_ok=True)
            os.makedirs(os.path.join(args.dst, 'alignment_attribs'), exist_ok=True)
            try:
                for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                    aligned_face_path = os.path.join(args.dst, f'align-{img_name}')
                    image_align_68(raw_img_path, aligned_face_path, face_landmarks, args.output_size, args.transform_size,
                                   args.no_padding)
            except TypeError as e:
                print(f'{e}: skipping file {raw_img_path}')
    else:
        aligned_img_path = os.path.join(args.src, 'aligned_inverted_images')
        for img_name in tqdm.tqdm(get_my_file_list(args.process_id, args.total_processes, aligned_img_path)):
            raw_img_path = os.path.join(aligned_img_path, img_name)
            dst_file = os.path.join(args.dst, f'dalign-{img_name}')
            # import ipdb; ipdb.set_trace()

            original_transform_file = os.path.join(args.src, 'alignment_attribs', img_name[:-3] + 'npz')

            # overlay_file = None
            overlay_file = os.path.join(args.overlay_dir, img_name[6:])
            reverse_transform(raw_img_path, dst_file, overlay_file, original_transform_file)

