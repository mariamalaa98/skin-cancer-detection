import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class MelanomaDataset(Dataset):
    def __init__(self, dataset_path, classes_value_dict: dict, transform=None):

        self.data = []  # [[img_full_path,img_value]]
        for class_name, class_value in classes_value_dict.items():
            imgs = os.listdir(dataset_path + "/" + class_name)
            for img_name in imgs:
                full_path = f"{dataset_path}/{class_name}/{img_name}"
                self.data.append([full_path, class_value])

        self.data_size = len(self.data)
        self.transform = transform

    def __getitem__(self, idx):
        img = np.array(Image.open(self.data[idx][0]))
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor([self.data[idx][1]],dtype=torch.float)
    def __len__(self):
        return self.data_size