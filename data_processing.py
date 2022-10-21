import os
import  cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import PIL
import PIL.Image
import pathlib
import AugmentedAlzheimerDataset
from AugmentedAlzheimerDataset import MildDemented, MildDemented,NonDemented,VeryMildDemented

DATA_DIR = "./AugmentedAlzheimerDataset"
CATEGORIES = ["MildDemented","MildDemented","NonDemented","VeryMildDemented"]
IMG_SIZE = 50

data = []
data_used = 100


def data_creation():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR,category)
        class_enumeration = CATEGORIES.index(category)
        i = 1
        for img in os.listdir(path):

            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([img_array, class_enumeration])
            except Exception as e:
                pass
            i = i+1
            if i == data_used:
                break
    return data


data = data_creation()

