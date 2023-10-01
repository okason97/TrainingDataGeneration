# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_util.py

import os
import random

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder
from scipy import io
from PIL import ImageOps, Image
import torch
import torchvision.transforms as transforms
import h5py as h5
import numpy as np
import json
from scipy.stats import multivariate_normal
import cv2
from skimage.draw import line_aa

class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0] else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1] else np.random.randint(low=0, high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__

class PoseFolder(Dataset):
    def __init__(self, data_dir, shapes, skeleton, transform = None):
        self.data_dir = data_dir
        self.samples = self.make_dataset()
        self.shapes = shapes
        self.transform = transform
        self.skeleton = skeleton
        self.n_keypoints = 20 if self.skeleton else 21 

    def make_dataset(self):    
        instances = []
        _, class_to_idx = self.find_classes(self.data_dir)
        for target_class in sorted(class_to_idx.keys()):
            target_dir = os.path.join(self.data_dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    instances.append(os.path.join(root, fname))
        return instances

    def loader(self, path, shape):
        f = open(path)
        pose = json.load(f)
        f.close()
        if len(pose['people']) > 0 and (sum([pose['people'][0]['hand_right_keypoints_2d'][i] for i in range(2,len(pose['people'][0]['hand_right_keypoints_2d']),3)]) > 0 or 
                                        sum([pose['people'][0]['hand_left_keypoints_2d'][i] for i in range(2,len(pose['people'][0]['hand_left_keypoints_2d']),3)]) > 0):
            rh_keypoints = pose['people'][0]['hand_right_keypoints_2d']
            lh_keypoints = pose['people'][0]['hand_left_keypoints_2d']
            if sum([rh_keypoints[i] for i in range(2,len(rh_keypoints),3)]) >= sum([lh_keypoints[i] for i in range(2,len(lh_keypoints),3)]):
                return self.load_keypoints(rh_keypoints, shape)
            else:
                return self.load_keypoints(lh_keypoints, shape)
        else:
            return torch.zeros((self.n_keypoints, shape[1], shape[0]))

    def load_keypoints(self, keypoints, shape):
        if self.skeleton:
            sample = np.zeros((self.n_keypoints, shape[1], shape[0]))
            for i in range(5):
                x0 = np.clip(int(keypoints[0]), 0, shape[0]-1)
                y0 = np.clip(int(keypoints[1]), 0, shape[1]-1)
                for j in range(1,5):
                    x1 = np.clip(int(keypoints[(i*4+j)*3]), 0, shape[0]-1)
                    y1 = np.clip(int(keypoints[(i*4+j)*3+1]), 0, shape[1]-1)
                    rr, cc, val = line_aa(y0, x0, y1, x1)
                    sample[i*4+j-1, rr, cc] = val
                    x0 = x1
                    y0 = y1
            return torch.tensor(sample).float()
        else:
            x, y = np.mgrid[0:shape[0]:1, 0:shape[1]:1]
            pos = np.dstack((x, y))
            sample = torch.zeros((self.n_keypoints, shape[1], shape[0]))
            for i in range(self.n_keypoints):
                kx = shape[0]-keypoints[i*3]
                ky = keypoints[i*3+1]
                # confidence = np.clip(keypoints[i*3+2], 0, 1)
                # shape[1]-shape[1]*confidence+1 | shape[0]-shape[0]*confidence+1
                rv = multivariate_normal([kx, ky], [[4, 0.], [0., 4]])
                sample[i] = torch.tensor(np.rot90(rv.pdf(pos),3).copy()).float()
            return sample


    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.loader(self.samples[index], self.shapes[index])
        if self.transform is not None:
            sample = self.transform(sample)        
        return sample

class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            minEdge = min(img.shape[1:])
        else:
            minEdge = min(img.size)
        return transforms.functional.center_crop(img, minEdge)

    def __repr__(self):
        return self.__class__.__name__

class ResizeTensor(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, t):
        return torch.nn.functional.interpolate(t.unsqueeze(0), size=self.shape, mode='nearest').squeeze(0)

    def __repr__(self):
        return self.__class__.__name__

class Dataset_(Dataset):
    def __init__(self,
                 data_name,
                 data_dir,
                 train,
                 crop_long_edge=False,
                 resize_size=None,
                 random_flip=False,
                 normalize=True,
                 hdf5_path=None,
                 load_data_in_memory=False,
                 pose=False,
                 skeleton=False):
        super(Dataset_, self).__init__()
        self.data_name = data_name
        self.data_dir = data_dir
        self.train = train
        self.random_flip = random_flip
        self.normalize = normalize
        self.hdf5_path = hdf5_path
        self.load_data_in_memory = load_data_in_memory
        self.pose = pose
        self.skeleton = skeleton
        self.trsf_list = []
        self.pose_trsf_list = []

        if self.hdf5_path is None:
            if crop_long_edge:
                self.trsf_list += [CenterCropLongEdge()]
                self.pose_trsf_list += [CenterCropLongEdge()]
            if resize_size is not None:
                self.trsf_list += [transforms.Resize(resize_size, Image.LANCZOS)]
                self.pose_trsf_list += [ResizeTensor(resize_size)]
        else:
            self.trsf_list += [transforms.ToPILImage()]

        if self.random_flip:
            self.trsf_list += [transforms.RandomHorizontalFlip()]

        if self.normalize:
            self.trsf_list += [transforms.ToTensor()]
            self.trsf_list += [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        else:
            self.trsf_list += [transforms.PILToTensor()]

        self.trsf = transforms.Compose(self.trsf_list)

        self.pose_trsf = transforms.Compose(self.pose_trsf_list)

        self.load_dataset()

    def load_dataset(self):
        if self.hdf5_path is not None:
            with h5.File(self.hdf5_path, "r") as f:
                if self.pose:
                    data, labels, poses = f["imgs"], f["labels"], f["poses"]
                else:
                    data, labels = f["imgs"], f["labels"]
                self.num_dataset = data.shape[0]
                if self.load_data_in_memory:
                    print("Load {path} into memory.".format(path=self.hdf5_path))
                    self.data = data[:]
                    self.labels = labels[:]
                    if self.pose:
                        self.poses = poses[:]
            return

        if self.data_name == "CIFAR10":
            self.data = CIFAR10(root=self.data_dir, train=self.train, download=True)

        elif self.data_name == "CIFAR100":
            self.data = CIFAR100(root=self.data_dir, train=self.train, download=True)
        else:
            mode = "train" if self.train == True else "valid"
            root = os.path.join(self.data_dir, mode)
            self.data = ImageFolder(root=root)
            if self.pose:
                pose_root = os.path.join(self.data_dir, mode+'_poses')
                self.poses = PoseFolder(pose_root, [(sample[0].size[0], sample[0].size[1]) for sample in self.data], self.skeleton)    

    def _get_hdf5(self, index):
        with h5.File(self.hdf5_path, "r") as f:
            if self.pose:
                return f["imgs"][index], f["labels"][index], f["poses"][index]
            else:
                return f["imgs"][index], f["labels"][index]

    def __len__(self):
        if self.hdf5_path is None:
            num_dataset = len(self.data)
        else:
            num_dataset = self.num_dataset
        return num_dataset

    def __getitem__(self, index):
        if self.pose:
            if self.hdf5_path is None:
                img, label = self.data[index]
                poses = self.poses[index]
            else:
                if self.load_data_in_memory:
                    img, label, poses = self.data[index], self.labels[index], self.poses[index]
                else:
                    img, label, poses = self._get_hdf5(index)
            return self.trsf(img), int(label), self.pose_trsf(poses)
        else:
            if self.hdf5_path is None:
                img, label = self.data[index]
            else:
                if self.load_data_in_memory:
                    img, label = self.data[index], self.labels[index]
                else:
                    img, label = self._get_hdf5(index)
            return self.trsf(img), int(label)