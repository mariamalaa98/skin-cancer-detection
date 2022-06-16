import gdown as gdown
from PIL import Image
import os


def load_img(img_path, transform=None, expand_dims=False):
    img = Image.open(img_path)
    if transform is not None:
        img = transform(img)
        if expand_dims:
            img = img.unsqueeze(0)
    return img


def get_last_weights_path(weights_path):
    files = os.listdir(weights_path)
    files.sort()
    last_weight_file_name = files[-1]
    return f"{weights_path}/{last_weight_file_name}"


def load_model_weights(model):
    url="https://drive.google.com/u/0/uc?id=1a4wTJDaV8bsYiDhOIuOPULP7o2NzD-Zf&export=download"
    weights_path = os.path.abspath("model_weights/mel_classifier_weights.pt")
    if not os.path.exists(os.path.abspath("model_weights")):
        os.mkdir(os.path.abspath("model_weights"))
        gdown.download(url, weights_path, quiet=False)
    model.load_local_weights(weights_path)

        # -----------------------------------
    return True
