import os
import pandas as pd
import tqdm

video_dir = '/is/cluster/fast/scratch/ssanyal/video_gan/fashion_videos/UBC_fashion_smpl/train'
videos = os.listdir(video_dir)

for driven_id in tqdm.tqdm(range(100)):
    for driver_id in range(50):
        trn_df = {'A_paths': [], 'B_paths': []}
        for driven_frame_root_vid in videos[driven_id*5:driven_id*5 + 5]:
            driven_frame_img = os.path.join(video_dir, driven_frame_root_vid, 'frames/frame_00000.png')
            driven_frame_kp = os.path.join(video_dir, driven_frame_root_vid, 'kptsmpls/frame_00000.json')
            for driving_vid in videos[driver_id*10:driver_id*10 + 10]:
                curr_apath = {'ref': [driven_frame_img, ], 'gen': []}
                curr_bpath = {'ref': [driven_frame_kp, ], 'gen': []}
                # if driven_frame_root_vid == driving_vid:
                #     continue
                driving_vid_full_pth = os.path.join(video_dir, driving_vid, 'frames')
                for dfi in sorted(os.listdir(driving_vid_full_pth)):
                    driving_frame_img = os.path.join(video_dir, driving_vid, f'frames/{dfi}')
                    curr_apath['gen'].append(driving_frame_img)
                    driving_frame_kp = os.path.join(video_dir, driving_vid, f'kptsmpls/{dfi[:-4]}.json')
                    curr_bpath['gen'].append(driving_frame_kp)

                trn_df['A_paths'].append(curr_apath)
                trn_df['B_paths'].append(curr_bpath)


        train_dataframe = pd.DataFrame(data=trn_df)
        train_dataframe.to_csv(f'/is/cluster/fast/scratch/ssanyal/video_gan/fashion_videos/UBC_fashion_smpl/'
                               f'60vids_each/tain_list{driven_id}_{driver_id}.csv')
