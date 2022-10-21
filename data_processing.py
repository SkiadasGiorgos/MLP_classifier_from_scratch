import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import PIL
import PIL.Image
import pathlib
import AugmentedAlzheimerDataset
from AugmentedAlzheimerDataset import MildDemented, MildDemented, NonDemented, VeryMildDemented
import random

DATA_DIR = "./AugmentedAlzheimerDataset"
CATEGORIES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
IMG_SIZE = 40

data = []
data_used = 100


def data_creation():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_enumeration = CATEGORIES.index(category)
        i = 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # plt.imshow(img_array,cmap='gray')
                # plt.show()
                data.append([img_array, class_enumeration])
            except Exception as e:
                pass
            i = i + 1
            if i == data_used:
                break
    return data


data = data_creation()
def split_data(data):
    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split_data(data)


