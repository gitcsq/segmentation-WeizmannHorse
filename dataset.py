import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class horse_dataset(Dataset):
    def __init__(self, transform=None, mode='train', img_size=224):
        super(horse_dataset).__init__()
        self.img_size = img_size
        self.transform = transform if transform is not None else lambda x: x
        self.files = os.listdir('./data/horse/')
        data_num = len(self.files)
        np.random.shuffle(self.files)
        self.train_files = self.files[:int(0.85*data_num)]
        self.test_files = self.files[int(0.85*data_num):]
        self.mode = mode

    def __getitem__(self, item):

        if self.mode == 'train':
            img = Image.open(os.path.join('./data', 'horse', self.train_files[item])).resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR)
            img = np.array(img)
            img = self.transform(img)
            mask = Image.open(os.path.join('./data', 'mask', self.train_files[item])).resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR)
            mask = np.array(mask)
            mask = torch.LongTensor(mask)
            return img, mask
        else:
            img = Image.open(os.path.join('./data', 'horse', self.test_files[item])).resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR)
            img = np.array(img)
            img = self.transform(img)
            mask = Image.open(os.path.join('./data', 'mask', self.test_files[item])).resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR)
            mask = np.array(mask)
            mask = torch.LongTensor(mask)
            return img, mask, self.test_files[item]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_files)
        else:
            return len(self.test_files)


if __name__ == '__main__':
    a = horse_dataset()
    train_x = a.train_files
    a.__getitem__(0)

    pass
