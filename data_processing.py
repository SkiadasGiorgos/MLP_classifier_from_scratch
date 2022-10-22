import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier , NearestCentroid
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
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
IMG_SIZE = 100

data_used = 4000


def data_creation():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_enumeration = CATEGORIES.index(category)
        i = 1
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)).flatten()
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

def principal_components(x):
    pca = PCA(n_components=600)
    x_reduced = pca.fit_transform(x)
    pca_recovered = pca.inverse_transform(x_reduced)
    image = pca_recovered[1, :].reshape([IMG_SIZE, IMG_SIZE])

    plt.imshow(image, cmap='gray')
    plt.show()
    #
    # plt.grid()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_*100))
    # plt.xlabel("Number of components")
    # plt.ylabel("Explained variance")
    # plt.show()

    return x_reduced
def split_data(data):
    x = []
    y = []
    for features, label in data:
        x.append(features)
        y.append(label)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split_data(data)

# x_train_reduced = principal_components(x_train)
# print(len(x_train_reduced))

k_neighbors_model = KNeighborsClassifier(1)
k_neighbors_model.fit(x_train,y_train)
neighbors_accuracy = k_neighbors_model.score(x_test,y_test)
print(neighbors_accuracy)

nearest_centroid_model = NearestCentroid()
nearest_centroid_model.fit(x_train,y_train)
centroid_accuracy = nearest_centroid_model.score(x_test,y_test)
print(centroid_accuracy)