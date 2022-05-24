from functools import partial

import torch
from torch.hub import load_state_dict_from_url
from torchvision.models import EfficientNet
from torch import nn, load, save
from torchvision.models.efficientnet import MBConvConfig, model_urls


class SkinCancerModel(EfficientNet):

    def __init__(self, download_weights, version, **kwargs):
        progress = True
        args = efficientnet_args(version)
        super(SkinCancerModel, self).__init__(*args, **kwargs)
        if download_weights:
            if model_urls.get(version, None) is None:
                raise ValueError(f"No checkpoint is available for model type {version}")
            state_dict = load_state_dict_from_url(model_urls[version], progress=progress)
            self.load_state_dict(state_dict)

        # Change Full connected layer
        self.classifier = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(1280, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.25),
                                        nn.Linear(256, 1), nn.Sigmoid())

    def load_local_weights(self, path):
        state_dict = load(path)
        self.load_state_dict(state_dict)

    def save_weights(self, path):
        state_dict = self.state_dict()
        save(state_dict, path)


def get_inverted_residual_setting(width_mult, depth_mult):
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1)]
    return inverted_residual_setting


def efficientnet_args(version):
    if version == "efficientnet_b1":
        width_mult = 1.0
        depth_mult = 1.1
        dropout = 0.2
    elif version == "efficientnet_b0":
        width_mult = 0.1
        depth_mult = 1.0
        dropout = 0.2
    elif version == "efficientnet_b4":
        width_mult = 1.4
        depth_mult = 1.8
        dropout = 0.4
    else:
        raise ValueError(f"invalid version of efficientnet '{version}' ")
    return get_inverted_residual_setting(width_mult, depth_mult), dropout


class MelanomaClassifier(nn.Module):
    def __init__(self, patient_data_size):
        super().__init__()
        self.features = SkinCancerModel(True, "efficientnet_b1")
        self.features.load_local_weights("weights.pt")
        self.features.classifier = nn.Sequential(self.features.classifier[0], self.features.classifier[1],
                                                 self.features.classifier[2])

        input_size = self.features.classifier[-2].out_features + patient_data_size
        self.classifier = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(inplace=True),nn.Dropout(0.3), nn.Linear(64, 1),
                                        nn.Sigmoid())

    def forward(self, image, data):
        features = self.features(image)
        input_vector = torch.cat([features, data], dim=1)
        output=self.classifier(input_vector)

        return output

    def load_local_weights(self, path):
        state_dict = load(path)
        self.load_state_dict(state_dict)

    def save_weights(self, path):
        state_dict = self.state_dict()
        save(state_dict, path)
