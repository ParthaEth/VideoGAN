import os
import numpy as np
import tqdm
root_dir = '/is/cluster/fast/pghosh/datasets/bouncing_sq/init_cond'

labels = []
for file_id in tqdm.tqdm(range(70_000)):
    labels.append(np.load(os.path.join(root_dir, f'{file_id:05d}.npy')))

labels = np.stack(labels)
# import ipdb; ipdb.set_trace()
np.save(os.path.join(root_dir, '../vids/labels.npy'), labels)