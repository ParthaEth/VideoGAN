import copy
import os
import numpy as np


def print_nested_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + f'{key}:')
            print_nested_dict(value, indent + 1)
        else:
            print('\t' * indent + f'{key}: {value}')


def print_latex_lines(result_dict):
    for method in result_dict:  # one line each
        lat_line = f'{method} & '
        for dataset in ['FFHQ_ALL', 'FASHION', 'UCF']:  # ffhq, fashion, skytimelapse, ucf
            for inp_frame_num in ['3 frames', '8 frames']:
                for metric in ['accum_ssim', 'accum_psnr']:
                    lat_line += f'{result_dict[method][inp_frame_num][dataset][metric]:.2f} &'
        lat_line = lat_line[:-1]
        print(f'{lat_line} \\\\')


# root_dir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/interpolate'
root_dir = '/is/cluster/fast/pghosh/ouputs/video_gan_runs/single_vid_over_fitting/webvid-flowers/interpolate'
result_dict = {}
data_set_2_file_name_start = {'UCF': ['clips_test', 'clips_train'],
                              'FASHION': ['fasion_video', ],
                              'FFHQ_10M': ['ffhq_X_10_good_motions', ],
                              'FFHQ_ALL': ['ffhq_X_celebv_hq', ],
                              'SKY_TIMELAPSE': ['train_clips', ],
                              'WEBVID_FLOWERS': ['flower_train_vids', ]}


def get_dataset_name(file_name):
    for dataset, prefixes in data_set_2_file_name_start.items():
        for prefix in prefixes:
            if file_name.startswith(prefix):
                return dataset
    return None  # Return None if no matching dataset is found


for method in os.listdir(root_dir):
    # print(f'{method}')
    dataset_dict = {}
    for frame_count in os.listdir(os.path.join(root_dir, method)):
        # print(frame_count)
        accum_dict = {'accum_psnr': 0, 'accum_ssim': 0, 'num_files': 0}
        stats_dict = {}
        for npy_file in os.listdir(os.path.join(root_dir, method, frame_count)):
            name_curr_file = os.path.join(root_dir, method, frame_count, npy_file)
            if npy_file.endswith('.npz'):
                data_set_name = get_dataset_name(npy_file)
                if stats_dict.get(data_set_name, None) is None:
                    stats_dict[data_set_name] = copy.deepcopy(accum_dict)
                dat_tis_file = np.load(name_curr_file)
                # print(f'psnr:{dat_tis_file["inpaint_psnr"]}, SSIM: {dat_tis_file["inpaint_ssim"]}')
                stats_dict[data_set_name]['accum_psnr'] += dat_tis_file["inpaint_psnr"]
                stats_dict[data_set_name]['accum_ssim'] += dat_tis_file["inpaint_ssim"]
                stats_dict[data_set_name]['num_files'] += 1

        for data_set in stats_dict:
            stats_dict[data_set]['accum_psnr'] = \
                round(stats_dict[data_set]['accum_psnr'] / stats_dict[data_set]['num_files'], 2)
            stats_dict[data_set]['accum_ssim'] = \
                round(stats_dict[data_set]['accum_ssim'] / stats_dict[data_set]['num_files'], 2)
        dataset_dict[frame_count] = copy.deepcopy(stats_dict)
        # print(stats_dict)
    result_dict[method] = copy.deepcopy(dataset_dict)

print(print_nested_dict(result_dict))
print_latex_lines(result_dict)