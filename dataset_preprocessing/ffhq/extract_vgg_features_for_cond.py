import torch
import torchvision.transforms
from torchvision.models import vgg16
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import os
from PIL import Image


class FeatureModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = None

        if model_name.lower() == 'vgg16':
            self.model = vgg16(pretrained=True)
            self.model.classifier = self.model.classifier[:-6]
        else:
            raise NotImplementedError(f'Model {model_name} not implemented!')

    def forward(self, inp):
        out = self.model(inp)
        return out


class FFHQDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith('.png'):
                self.img_labels.append(filename)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return label, image


def get_dataloader(data_loader_name):
    if data_loader_name.lower() == 'random_5x3x224x224':
        data_loader = [(['1.png', '2.png', '3.png', '4.png', '5.png', ],
                        torch.rand((5, 3, 224, 224)))]
    elif data_loader_name.lower() == 'ffhq':
        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.CenterCrop(224),
                                                     torchvision.transforms.Normalize(mean=normalization_mean,
                                                                                      std=normalization_std)])
        data_loader = torch.utils.data.DataLoader(FFHQDataset('/is/cluster/fast/pghosh/datasets/ffhq/256X256',
                                                              transform=transforms),
                                                  batch_size=16, shuffle=False, num_workers=8)
    else:
        raise NotImplementedError(f'Dataloader {data_loader_name} not implemented!')
    return data_loader


def run_feature_extraction(model_name, data_loader_name,
                           condition_file_path=None):
    feature_model = FeatureModel(model_name=model_name)
    data_loader = get_dataloader(data_loader_name=data_loader_name)

    if condition_file_path is not None:
        condition_data = read_json_condition_file(condition_file_path=condition_file_path)
        condition_dict = convert_json_condition_data(condition_data=condition_data)

    for filenames, batch in tqdm(data_loader):
        feature_out = feature_model(batch)
        if condition_file_path is not None:
            modify_condition_data(condition_dict=condition_dict, filenames=filenames,
                                  new_features=feature_out)
    return condition_dict


def write_formatted_json(condition_dict, output_file_path):
    fh = open(output_file_path, 'w')
    json.dump({'labels': list(condition_dict.items())}, fh, indent=4)
    fh.close()


def read_json_condition_file(condition_file_path):
    with open(condition_file_path, 'r') as fh:
        fh_data = fh.read()
        condition_data = json.loads(fh_data)
    return condition_data


def convert_json_condition_data(condition_data):
    condition_data_dict = dict(condition_data['labels'])
    return condition_data_dict


def modify_condition_data(condition_dict, filenames, new_features):
    for idx, filename in enumerate(filenames):
        if filename in condition_dict:
            existing_feature = [0.9827,  0.0000, -0.1852,  0.5000,  0.0000, -1.0000,  0.0000,  0.0000,
                                -0.1852,  0.0000, -0.9827,  2.6533,  0.0000,  0.0000,  0.0000,  1.0000,
                                4.2634,  0.0000,  0.5000,  0.0000,  4.2634,  0.5000,  0.0000,  0.0000,
                                1.0000]
        # condition_dict[filename]
            processed_new_feature = new_features[idx].detach().cpu().numpy().tolist()
            extended_feature = existing_feature + processed_new_feature
            condition_dict[filename] = extended_feature
        else:
            print(f'filename {filename} not in condition_dict')


def sanity_check_condition_dict():
    pass


condition_dict = run_feature_extraction(
    model_name='vgg16',
    data_loader_name='FFHQ',
    condition_file_path='/is/cluster/fast/pghosh/datasets/ffhq/256X256/dataset.json')
write_formatted_json(condition_dict=condition_dict,
                     output_file_path='/is/cluster/fast/pghosh/datasets/ffhq/vgg_features_with_cam.json')
