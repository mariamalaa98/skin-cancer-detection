from functools import partial

import torch
from torch import nn, load, save, Tensor

from torchvision.models import convnext_tiny


class SkinCancerModel(nn.Module):

    def __init__(self, download_weights=False):
        super().__init__()
        convnext_tiny_model = convnext_tiny(pretrained=download_weights)
        self.features = convnext_tiny_model.features
        # Change Full connected layer
        self.classifier = convnext_tiny_model.classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier[-1] = nn.Linear(self.classifier[-1].in_features, 256)
        self.classifier.append(nn.Dropout())
        self.classifier.append(nn.Linear(256, 1))
        self.classifier.append(nn.Sigmoid())

    def load_local_weights(self, path, cuda_weights=False):
        if cuda_weights:
            device = torch.device('cpu')
            state_dict = load(path, map_location=device)
        else:
            state_dict = load(path)
        self.load_state_dict(state_dict)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def save_weights(self, path):
        state_dict = self.state_dict()
        save(state_dict, path)


class MelanomaClassifier(nn.Module):
    def __init__(self, patient_data_size, feature_model_weights):
        super().__init__()
        self.features = SkinCancerModel(True)
        self.features.load_local_weights(feature_model_weights, True)
        self.features.classifier = nn.Sequential(self.features.classifier[0], self.features.classifier[1],
                                                 self.features.classifier[2])

        input_size = self.features.classifier[-1].out_features + patient_data_size
        self.classifier = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(inplace=True), nn.Dropout(0.35),
                                        nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(32, 1),
                                        nn.Sigmoid())

    def forward(self, image, data):
        features = self.features(image)
        input_vector = torch.cat([features, data], dim=1)
        output = self.classifier(input_vector)

        return output

    def load_local_weights(self, path):
        state_dict = load(path)
        self.load_state_dict(state_dict)

    def save_weights(self, path):
        state_dict = self.state_dict()
        save(state_dict, path)
