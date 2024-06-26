import numpy as np
import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def split_sorted_frames(sorted_frames, sub_smpl_factor):
    sequences = []
    for i in range(sub_smpl_factor):
        sequence = sorted_frames[i::sub_smpl_factor]  # Use list slicing to select every nth element
        sequences.append(sequence)
    return sequences


def make_dataset(dir, nframes, class_to_idx, max_clips_per_vid, frame_offset, subsample_factor):
    images = []
    n_video = 0
    n_clip = 0
    for target in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir,target))==True:
            n_video +=1
            # eg: dir + '/rM7aPu9WV2Q'
            subfolder_path = os.path.join(dir, target) 
            for subsubfold in sorted(os.listdir(subfolder_path) ):
                if os.path.isdir(os.path.join(subfolder_path, subsubfold)):
                    # eg: dir + '/rM7aPu9WV2Q/1'
                    n_clip += 1
                    subsubfolder_path = os.path.join(subfolder_path, subsubfold) 
                    
                    item_frames = []
                    i = 1
                    clips_frm_this_vid = 0
                    sorted_frames = sorted([imf for imf in os.listdir(subsubfolder_path) if is_image_file(imf)])
                    if subsample_factor > 1:
                        frame_seqs = split_sorted_frames(sorted_frames, subsample_factor)
                    else:
                        frame_seqs = [sorted_frames, ]
                    for frame_seq in frame_seqs:
                        if frame_offset.lower() == 'random':
                            if len(frame_seq) < nframes:
                                continue
                            offset = np.random.randint(0, len(frame_seq) - nframes)
                        elif frame_offset.lower() == 'none':
                            offset = 0
                        else:
                            raise ValueError(f'unrecognized offset mode: {frame_offset}')
                        for fi in frame_seq[offset:]:
                            file_name = fi
                            # eg: dir + '/rM7aPu9WV2Q/1/rM7aPu9WV2Q_frames_00086552.jpg'
                            file_path = os.path.join(subsubfolder_path, file_name)
                            item = (file_path, class_to_idx[target])
                            item_frames.append(item)
                            if i % nframes == 0 and i > 0:
                                images.append(item_frames) # item_frames is a list containing n frames.
                                item_frames = []
                                clips_frm_this_vid += 1
                            if max_clips_per_vid <= clips_frm_this_vid:
                                break
                            i = i+1
    print('number of long videos:')
    print(n_video)
    print('number of short videos')
    print(n_clip)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    '''
    return pil_loader(path)


class VideoFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, nframes,  transform=None, target_transform=None,
                 loader=default_loader, max_clips_per_vid=None, frame_offset='none', subsample_factor=1):
        classes, class_to_idx = find_classes(root)
        self.max_clips_per_vid = np.inf if max_clips_per_vid is None else max_clips_per_vid
        self.frame_offset = 'none' if frame_offset is None else frame_offset
        self.subsample_factor = 1 if subsample_factor is None else subsample_factor

        imgs = make_dataset(root, nframes,  class_to_idx, self.max_clips_per_vid, self.frame_offset,
                            self.subsample_factor)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + 
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.nframes = nframes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # clip is a list of 32 frames 
        clip = self.imgs[index] 
        img_clip = []
        i = 0
        for frame in clip:
            path, target = frame
            img = self.loader(path) 
            i = i+1
            if self.transform is not None:
                img = self.transform(img)
            img = img.view(img.size(0),1, img.size(1), img.size(2))
            img_clip.append(img)
        img_frames = torch.cat(img_clip, 1) 
        return img_frames, target

    def __len__(self):
        return len(self.imgs)
