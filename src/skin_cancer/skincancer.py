import numpy as np
import torch
from torchvision.transforms import transforms

from . import utils
from .model import MelanomaClassifier

global melanoma_classifier
if "melanoma_classifier" not in globals():
    melanoma_classifier = MelanomaClassifier(24)
    utils.load_model_weights(melanoma_classifier)


def predict_melanoma(img_path, age, gender, location, threshold=0.48):
    """
    predict the given skin image is a melanoma or not
    :param img_path: (str) img full path
    :param age: (int) patient age
    :param gender: (str) ['male','female']
    :param location: (str) location of the photo
    :param threshold: (float) classification threshold
    :return: pair (prediction result boolean , probability of being melanoma )
    """
    location_dict = {"abdomen": 1, "acral": 2, "anterior torso": 3, "back": 4,
                     "chest": 5, "ear": 6, "face": 7, "foot": 8,
                     "genital": 9, "hand": 10, "head/neck": 11, "lateral torso": 12,
                     "lower extremity": 13, "neck": 14, "oral/genital": 15,
                     "palms/soles": 16, "posterior torso": 17, "scalp": 18,
                     "torso": 19, "trunk": 20, "upper extremity": 21}
    gender_dict = {"female": 22, "male": 23}

    global melanoma_classifier

    if "melanoma_classifier" not in globals():
        melanoma_classifier = MelanomaClassifier(24)
        utils.load_model_weights(melanoma_classifier)

    if gender not in gender_dict:
        raise ValueError("invalid gender")

    if location not in location_dict:
        raise ValueError("invalid location")
    transform = transforms.Compose(
        [transforms.Resize((240, 240)), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    try:
        img = utils.load_img(img_path, transform, expand_dims=True)
    except:
        raise FileNotFoundError(f"can't load the image in path {img_path}")

    patient_data = np.zeros(24)
    patient_data[0] = age
    patient_data[location_dict[location]] = 1
    patient_data[gender_dict[gender]] = 1
    patient_data = torch.tensor(patient_data, dtype=torch.float)
    patient_data = patient_data.unsqueeze(0)
    try:

        output = melanoma_classifier.predict(img, patient_data)
    except:
        raise Exception("error loading data from melanoma classifier model")

    if output < threshold:
        return False, output
    else:
        return True, output
