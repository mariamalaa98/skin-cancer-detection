import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision.transforms.functional import normalize


def data_preprocessing(raw_data_path, prep_data_path, transform):
    imgs_list = os.listdir(raw_data_path)
    for img_name in imgs_list:
        img_path = raw_data_path + '/' + img_name
        image = Image.open(img_path)
        cropped_img = transform(image)
        cropped_img.save(f"{prep_data_path}/{img_name}")


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


class MelanomaImageDataset(Dataset):
    def __init__(self, csv_path, dataset_path, transform=None):
        df = pd.read_csv(csv_path, index_col=False)
        self.data = df.to_numpy()  # [[img_path,img_value]]
        self.data_size = len(df)
        self.transform = transform
        self.dataset_path = dataset_path

    def __getitem__(self, idx):
        path = f"{self.dataset_path}/{self.data[idx][0]}.jpg"
        img = np.array(Image.open(path))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([self.data[idx][1]], dtype=torch.float)

    def __len__(self):
        return self.data_size


class MelanomaHam1000Dataset(Dataset):

    def __init__(self, csv_path, dataset_path, transform=None):
        df = pd.read_csv(csv_path, index_col=False)
        self.data = df.to_numpy()  # [[img_path,img_value]]
        self.data_size = len(df)
        self.transform = transform
        self.dataset_path = dataset_path

    def __getitem__(self, idx):
        path = f"{self.dataset_path}/{self.data[idx][0]}.jpg"
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        image = image.float()
        patient_data = self.data[idx][2:].astype(float)

        label = self.data[idx][1]

        return image, torch.tensor(patient_data,dtype=torch.float), torch.tensor(label,
                                                                   dtype=torch.float)

    def __len__(self):
        return self.data_size


def ham_data_csv_preprocessing(csv_path):
    data = pd.read_csv(csv_path, index_col=False)
    del data['lesion_id']
    del data['dx_type']
    del data['dataset']
    for i in range(data.shape[0]):
        if data.iloc[i][1] == "mel":
            data.at[i, 'dx'] = 1
        else:
            data.at[i, 'dx'] = 0

    # Drop sex unknown
    idx_to_remove = set()
    for i in range(len(data)):
        if data.iloc[i][3] == "unknown":
            idx_to_remove.add(i)

    for i in range(len(data)):
        if data.iloc[i][4] == "unknown":
            idx_to_remove.add(i)
    data = data.drop(index=list(idx_to_remove))

    dummy_cols = ['localization', 'sex']
    for col in dummy_cols:
        dummy = pd.get_dummies(data[col], prefix=col, drop_first=False)
        data = pd.concat([data, dummy], axis=1)

    data.drop(dummy_cols, axis=1, inplace=True)
    return data
