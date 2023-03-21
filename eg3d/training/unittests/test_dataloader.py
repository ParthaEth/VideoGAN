import sys
sys.path.append('../../')
import tqdm
import torch
from training.dataset import VideoFolderDataset


training_set = VideoFolderDataset(path='/is/cluster/fast/pghosh/datasets/ffhq/256X256_zoom_vid', return_video=True)
training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, batch_size=4, num_workers=8,
                                                         worker_init_fn=training_set.worker_init_fn))

for img, label in tqdm.tqdm(training_set_iterator):
    pass