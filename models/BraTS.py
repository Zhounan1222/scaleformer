import os
import pickle
import random

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        Max = np.max(image)
        Min = np.min(image)
        image = (image - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)

        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]

        return {'image': image, 'label': label}


class Extend_label(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        net = (label == 1)
        ed = (label == 2)
        et = (label == 4)

        tc = np.logical_or(et, net)
        wt = np.logical_or(tc, ed)
        label1 = np.stack([net, ed, et]) 
        label2 = np.stack([et, tc, wt])
        et_present = 1 if np.sum(et) > 0 else 0
        return {'image': image, 'label1': label1, 'label2': label2, 'et_present': et_present}


class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']
        
        scale_factor = 1. 
        shift_factor = 0. 
        if random.random() < 0.8:
            scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, image.shape[1], 1, image.shape[-1]])
        if random.random() < 0.2:
            shift_factor = np.random.uniform(-factor, factor, size=[1, image.shape[1], 1, image.shape[-1]])

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}


class Random_rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        angle = round(np.random.uniform(-10, 10), 2)
        image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)

        return {'image': image, 'label': label}


class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label1 = sample['label1']
        label2 = sample['label2']
        label1 = np.ascontiguousarray(label1)
        label2 = np.ascontiguousarray(label2)

        sample['image'] = torch.from_numpy(image).float()
        sample['label1'] = torch.from_numpy(label1).long()
        sample['label2'] = torch.from_numpy(label2).long()

        return sample

class Crop_no_overlap(object):
    def __init__(self, size=128):
        self.patch_size = 128
    
    def __call__(self, image):
        
        pass
    


def transform(sample):
    trans = transforms.Compose([
        Pad(),
        # Random_rotate(),  # time-consuming
        Random_Crop(),
        Random_Flip(),
        Random_intencity_shift(),
        Extend_label(),
        ToTensor(),
    ])

    return trans(sample)


def transform_valid(sample):
    trans = transforms.Compose([
        Pad(),
        # MaxMinNormalization(),
        # Crop_no_overlap(),
        Extend_label(),
        ToTensor(),
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, list_file, root='', mode='train'):
        self.lines = []
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line, name + '_')
                # path = os.path.join(root,line+'_')
                paths.append(path)
                self.lines.append(line)
        self.mode = mode
        self.names = names
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if self.mode == 'train':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform(sample)
            return sample
        elif self.mode == 'valid':
            image, label = pkload(path + 'data_f32b0.pkl')
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            sample['patient_idx'] = os.path.basename(path)[:-1]
            return sample
        else:
            image = pkload(path + 'data_f32b0.pkl')
            affine = nib.load(path + 'flair.nii.gz').affine
            image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
            image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
            image = torch.from_numpy(image).float()
            return image, self.names[item], affine

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]



if __name__ == "__main__":
    list_file = "/workspace/multimodal/datasets/BraTs2019/training/train_split.txt"
    train_root = "/workspace/multimodal/datasets/BraTs2019/training"
    dataset = BraTS(list_file, train_root, mode='train')
    dataset.__getitem__(0)
