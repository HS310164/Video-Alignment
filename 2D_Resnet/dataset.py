import glob
import json
import os

import torch.utils.data as data
from PIL import Image


class TransDataset(data.Dataset):
    def __init__(self, im_paths, transform=None):
        self.im_paths = im_paths
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        if self.transform:
            im = self.transform(im)
        return im


class TrainDataset(data.Dataset):
    def __init__(self, im_paths, transform=None):
        dirs = sorted(glob.glob(os.path.join(im_paths, '*')))
        self.im_paths = []
        self.label_dict = {}
        for idx, i in enumerate(dirs):
            if not os.path.isdir(i):
                continue
            labelname = os.path.basename(i)
            self.label_dict[labelname] = idx
            videos = glob.glob(os.path.join(i, '*'))
            for j in videos:
                d = glob.glob(os.path.join(j, '*.jpg'))
                for p in d:
                    self.im_paths.append(p)
        self.transform = transform
        with open('label.json', 'w') as f:
            json.dump(self.label_dict, f)

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        im = pil_loader(self.im_paths[index])
        label = os.path.dirname(os.path.dirname(self.im_paths[index]))
        label = os.path.basename(label)
        label = self.label_dict[label]
        if self.transform:
            im = self.transform(im)
        return im, label


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
