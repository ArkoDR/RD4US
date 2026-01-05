import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def get_data_transforms(size):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
    ])
    return data_transforms


def get_pic_pair(filepath, transform):
    a_path = glob.glob(os.path.join(filepath, "*_a.png"))
    k_path = glob.glob(os.path.join(filepath, "*_k.png"))
    e_path = glob.glob(os.path.join(filepath, "*_e.png"))
    a = Image.open(a_path[0])
    k = Image.open(k_path[0])
    e = Image.open(e_path[0])
    a = transform(a)
    k = transform(k)
    e = transform(e)
    
    return a, e, k


class USDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode = "train", size = 256):
        self.mode = mode
        self.transform = get_data_transforms(size)
        self.normal = glob.glob(os.path.join(root, mode, "normal", "*"))
        self.label_normal = np.zeros(len(self.normal))
        self.abnormal = []
        if mode == "test":
            self.abnormal = glob.glob(os.path.join(root, mode, "abnormal", "*"))
        self.label_abnormal = np.ones(len(self.abnormal))

        self.data = self.normal + self.abnormal
        self.label = np.concatenate((self.label_normal, self.label_abnormal), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pics_a, pics_e, pics_k = get_pic_pair(self.data[idx], self.transform, self.mode)

        return pics_a, pics_e, pics_k, self.label[idx]