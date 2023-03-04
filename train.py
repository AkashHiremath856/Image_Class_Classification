import pickle
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from bing_image_downloader import downloader
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import shutil


def get_images(images, total=10):
    print('Getting Image')
    # Training Images
    shutil.rmtree('Images', ignore_errors=True)
    for _ in images:
        downloader.download(_, limit=total,
                            adult_filter_off=True, output_dir='Images')


def tranform_image():
    target = []
    images = []
    flatten_data = []

    DATA_DIR = 'Images/'
    CATEGORIES = os.listdir(DATA_DIR)
    CATEGORIES

    for categories in CATEGORIES:
        class_num = CATEGORIES.index(categories)  # label encoding
        path = os.path.join(DATA_DIR, categories)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = resize(img_array, (150, 150, 3))
            flatten_data.append(img_resized.flatten())  # Normalizes values
            images.append(img_resized)
            target.append(class_num)

    flatten_data = np.array(flatten_data)
    target = np.array(target)
    images = np.array(images)

    return flatten_data, target


def train_model_():
    flatten_data, target = tranform_image()

    x_train, x_test, y_train, y_test = train_test_split(
        flatten_data, target, test_size=0.3, random_state=20)

    grid_params = [
        {'C': [1.0, 1.00, 1.000], 'kernel':['linear']},
        {'C': [10, 100, 100], 'gamma':[0.001, 0.0001], 'kernel':['rbf']}
    ]

    # train using SVC
    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, grid_params)
    clf.fit(x_train, y_train)

    # Export model
    pickle.dump(clf, open('img_model.pkl', 'wb'))
