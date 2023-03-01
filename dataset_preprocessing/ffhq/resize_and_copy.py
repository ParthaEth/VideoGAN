from PIL import Image
import os
import tqdm


src_dir = '/is/cluster/fast/scratch/datasets/ffhq/images1024x1024/'
dest_dir = '/is/cluster/fast/pghosh/datasets/ffhq/256X256'
dataset_description = os.path.join(dest_dir, 'dataset.json')
dest_resolution = 256

os.makedirs(dest_dir, exist_ok=True)

max_count = 70000
curr_count = 0
file_object = open(dataset_description, 'a')
file_object.write('{\n')
file_object.write('    "labels": [\n')

files = os.listdir(src_dir)
num_files = len(files)

for curr_file in tqdm.tqdm(files):
    if curr_file.endswith('.jpg') or curr_file.endswith('.png'):
        curr_img = Image.open(os.path.join(src_dir, curr_file))
        curr_img = curr_img.resize((dest_resolution, dest_resolution))
        curr_img.save(os.path.join(dest_dir, curr_file))
        curr_count += 1
        line_this_img = f'        ["{curr_file}",[0,1.0,1.0,1.0,1.0,' \
                                           '1.0,1.0,1.0,1.0,1.0,' \
                                           '1.0,1.0,1.0,1.0,1.0,' \
                                           '1.0,1.0,1.0,1.0,1.0,' \
                                           '1.0,1.0,1.0,1.0,1.0]]'
        # import ipdb; ipdb.set_trace()
        if curr_count == num_files or curr_count == max_count:
            line_this_img += '\n'
        else:
            line_this_img += ',\n'

        file_object.write(line_this_img)

    if curr_count >= max_count:
        break

file_object.write('    ]\n')
file_object.write('}\n')
file_object.close()