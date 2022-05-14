import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms.functional import normalize


def data_preprocessing(raw_data_path, prep_data_path, transform):
    raw_classes_list = os.listdir(raw_data_path)
    for class_name in raw_classes_list:
        class_path = raw_data_path + class_name
        imgs_list = os.listdir(class_path)
        if not os.path.exists(f"{prep_data_path}/{class_name}"):
            os.mkdir(f"{prep_data_path}/{class_name}")
        for img_name in imgs_list:
            img_path = class_path + '/' + img_name
            image = Image.open(img_path)
            cropped_img = transform(image)
            cropped_img.save(f"{prep_data_path}/{class_name}/{img_name}")


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

        return img, torch.tensor([self.data[idx][1]], dtype=torch.float)

    def __len__(self):
        return self.data_size
class Normalize(torch.nn.Module):

    def forward(self, img):
        t_mean = torch.mean(img, dim=[1, 2])
        t_std = torch.std(img, dim=[1, 2])
        return normalize(img, t_mean.__array__(), t_std.__array__())

    def __init__(self):
        super().__init__()
