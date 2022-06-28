import math
import os
import shutil
import requests
import torch
import numpy as np
import pandas as pd
import imageio
from skimage import transform
import sklearn.utils
from torch.utils.data import Dataset
import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import json
import os
import pandas as pd
import pickle
from tqdm import tqdm

import random
import torch
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

def seed_all(seed=None):
    """Seed every type of random used by the SRGAN."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        seed = int(time.time())
    torch.manual_seed(seed)

def unison_shuffled_copies(a, b):
    """Shuffles two numpy arrays together."""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def gen_data_linear(n_instance):
    a = np.random.normal(3, 3, n_instance)
    samples = int(n_instance / 2)
    X = np.hstack((np.random.normal(4, 3, samples), np.random.normal(4, 3, samples)))
    y = np.hstack((X[:samples] + a[:samples], X[samples:] + a[samples:]))
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def gen_data_heteroscedastic(n_instance):
    X = np.random.normal(0, 1, n_instance)
    b = (0.001 + 0.5 * np.abs(X)) * np.random.normal(1, 1, n_instance)
    y = X + b
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def gen_data_multimodal(n_instance):
    x = np.random.rand(int(n_instance / 2), 1)
    y1 = np.ones((int(n_instance / 2), 1))
    y2 = np.ones((int(n_instance / 2), 1))
    y1[x < 0.4] = 1.2 * x[x < 0.4] + 0.2 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y2[x < 0.4] = x[x < 0.4] + 0.6 + 0.03 * np.random.randn(np.sum(x < 0.4))
    y1[np.logical_and(x >= 0.4, x < 0.6)] = 0.5 * x[np.logical_and(x >= 0.4, x < 0.6)] + 0.01 * np.random.randn(
        np.sum(np.logical_and(x >= 0.4, x < 0.6)))
    y2[np.logical_and(x >= 0.4, x < 0.6)] = 0.6 * x[np.logical_and(x >= 0.4, x < 0.6)] + 0.01 * np.random.randn(
        np.sum(np.logical_and(x >= 0.4, x < 0.6)))
    y1[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))
    y2[x >= 0.6] = 0.5 + 0.02 * np.random.randn(np.sum(x >= 0.6))
    y = np.array(np.vstack([y1, y2])[:, 0]).reshape((n_instance, 1))
    x = np.tile(x, (2, 1)) + 0.02 * np.random.randn(n_instance, 1)
    x = np.array(x[:, 0]).reshape((n_instance, 1))

    return x, y


def gen_data_exp(n_instance):
    z = np.random.normal(0, 1, n_instance)
    X = np.random.normal(0, 1, n_instance)
    y = X + np.exp(z)
    X = X.reshape((n_instance, 1))
    y = y.reshape((n_instance, 1))

    return X, y


def get_dataset(n_instance=1000, scenario="ailerons", seed=1):
    """
    Create regression data: y = x(1 + f(z)) + g(z)
    """

    if scenario == "CA-housing":
        housing = fetch_california_housing()

        X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=seed)

    elif scenario == "CA-housing-single":
        housing = fetch_california_housing()

        X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data[:, 0], housing.target,
                                                                      random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=seed)

        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        X_valid = X_valid.reshape(-1, 1)

    elif scenario == "ailerons":
        scaling = 10000

        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/Ailerons/ailerons.data", delimiter=',')
        X_train = my_data_train[:, 0:40]
        y_train = my_data_train[:, 40] * scaling

        my_data_test = np.genfromtxt(f"../FuzzyGAN/data/Ailerons/ailerons.test", delimiter=',')
        X_test_full = my_data_test[:, 0:40]
        y_test_full = my_data_test[:, 40] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "comp-activ":
        scaling = 1

        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/comp-activ/Prototask.data", delimiter=' ')

        X_train = my_data_train[0:4096, 0:21]
        y_train = my_data_train[0:4096, 21] * scaling

        X_test_full = my_data_train[4096:8192, 0:21]
        y_test_full = my_data_train[4096:8192, 21] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "pumadyn":
        scaling = 1000
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/pumadyn-32nm/Prototask.data", delimiter=' ')

        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        my_data_train_scaled = min_max_scaler.fit_transform((my_data_train[0:8192, 32] * scaling).reshape(-1, 1))

        X_train = my_data_train[0:4096, 0:32]
        y_train = my_data_train_scaled[0:4096]

        X_test_full = my_data_train[4096:8192, 0:32]
        y_test_full = my_data_train_scaled[4096:8192]

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "bank":
        scaling = 10
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/bank-32nm/Prototask.data", delimiter=' ')
        X_train = my_data_train[0:4096, 0:32]
        y_train = my_data_train[0:4096, 32] * scaling

        X_test_full = my_data_train[4096:8192, 0:32]
        y_test_full = my_data_train[4096:8192, 32] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "census-house":
        scaling = 10 ** -5
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/census-house/house-price-16H/Prototask.data",
                                      delimiter=' ')
        X_train = my_data_train[0:11392, 0:16]
        y_train = my_data_train[0:11392, 16] * scaling

        X_test_full = my_data_train[11392:22784, 0:16]
        y_test_full = my_data_train[11392:22784, 16] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "abalone":
        scaling = 1
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/abalone/Prototask.data", delimiter=' ')
        X_train = my_data_train[0:2089, 1:8]
        y_train = my_data_train[0:2089, 8] * scaling

        X_test_full = my_data_train[2089:4177, 1:8]
        y_test_full = my_data_train[2089:4177, 8] * scaling

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "driving":

        database_directory = f"../SRGAN/Steering Angle Database/Steering Angle Database/driving_dataset/driving_dataset/"
        seed_all(seed)
        dataset_path = database_directory
        # with open(os.path.join(self.dataset_path, 'data.txt')) as f:
        #meta = pd.read_csv(os.path.join(self.dataset_path, 'meta.pkl'), header=None, delimiter=" ")
        meta = pd.read_pickle(os.path.join(dataset_path, 'meta.pkl'))
        meta = sklearn.utils.shuffle(meta, random_state=seed)  # Shuffles only first axis.
        image_names = meta.iloc[:, 0].values
        angles = meta.iloc[:, 1].values
        # print(angles)
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        angles = min_max_scaler.fit_transform(angles.reshape(-1, 1)).reshape(-1)
        image_names = np.array(image_names)
        angles = np.array(angles, dtype=np.float32)
        # Force full batch sizes
        length = int(image_names.shape[0]/2)
        full_length = int(image_names.shape[0])
        X_train = []
        y_train = []
        X_test_full = []
        y_test_full = []
        for i in range(0,5000):
            image = np.load(os.path.join(dataset_path, image_names[i].replace('.jpg', '.npy')))
            image = tf.convert_to_tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            X_train.append(tf.reshape(image, [128,128,3]))
            angle = angles[i]
            angle = tf.convert_to_tensor(angle, dtype=tf.float32)
            y_train.append(angle)
        for j in range(5000,10000):
            image = np.load(os.path.join(dataset_path, image_names[j].replace('.jpg', '.npy')))
            image = tf.convert_to_tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            X_test_full.append(tf.reshape(image, [128,128,3]))
            angle = angles[j]
            angle = tf.convert_to_tensor(angle, dtype=tf.float32)
            y_test_full.append(angle)

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)
    elif scenario == "age":
        dataset_path = f"../SRGAN/imdb_wiki_data/imdb_preprocessed_128"
        with open(os.path.abspath(os.path.join(dataset_path, 'meta.json'))) as json_file:
            json_contents = json.load(json_file)
        image_names, ages = [], []
        for entry in json_contents:
            if isinstance(entry, dict):
                image_names.append(entry['image_name'])
                ages.append(entry['age'])
            else:
                image_name, age, gender = entry
                image_names.append(image_name)
                ages.append(age)
        image_names, ages = np.array(image_names), np.array(ages)
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        ages = min_max_scaler.fit_transform(ages.reshape(-1, 1)).reshape(-1)
        image_names = np.array(image_names)
        ages = np.array(ages, dtype=np.float32)
        length = int(image_names.shape[0]/2)
        full_length = int(image_names.shape[0])
        X_train = []
        y_train = []
        X_test_full = []
        y_test_full = []
        for i in range(0,5000):
            image_name = image_names[i]
            image = imageio.imread(os.path.join(dataset_path, image_name))
            image = image.transpose((2, 0, 1))
            image = tf.convert_to_tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            age = ages[i]
            age = tf.convert_to_tensor(age, dtype=tf.float32)
            X_train.append(tf.reshape(image, [128,128,3]))
            y_train.append(age)
        for j in range(5000,10000):
            image_name = image_names[j]
            image = imageio.imread(os.path.join(dataset_path, image_name))
            image = image.transpose((2, 0, 1))
            image = tf.convert_to_tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            age = ages[j]
            age = tf.convert_to_tensor(age, dtype=tf.float32)
            X_test_full.append(tf.reshape(image, [128,128,3]))
            y_test_full.append(age)

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)
    elif scenario == "driving_torch":

        database_directory = f"../SRGAN/Steering Angle Database/Steering Angle Database/driving_dataset/driving_dataset/"
        seed_all(seed)
        dataset_path = database_directory
        # with open(os.path.join(self.dataset_path, 'data.txt')) as f:
        #meta = pd.read_csv(os.path.join(self.dataset_path, 'meta.pkl'), header=None, delimiter=" ")
        meta = pd.read_pickle(os.path.join(dataset_path, 'meta.pkl'))
        meta = sklearn.utils.shuffle(meta, random_state=seed)  # Shuffles only first axis.
        image_names = meta.iloc[:, 0].values
        angles = meta.iloc[:, 1].values
        # print(angles)
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        angles = min_max_scaler.fit_transform(angles.reshape(-1, 1)).reshape(-1)
        image_names = np.array(image_names)
        angles = np.array(angles, dtype=np.float32)
        # Force full batch sizes
        length = int(image_names.shape[0]/2)
        full_length = int(image_names.shape[0])
        X_train = []
        y_train = []
        X_test_full = []
        y_test_full = []
        for i in range(0,5000):
            image = np.load(os.path.join(dataset_path, image_names[i].replace('.jpg', '.npy')))
            image = torch.tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            X_train.append(image.reshape(-1,128,128,3))
            angle = angles[i]
            angle = torch.tensor(angle, dtype=torch.float32)
            y_train.append(angle)
        for j in range(5000,10000):
            image = np.load(os.path.join(dataset_path, image_names[j].replace('.jpg', '.npy')))
            image = torch.tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            X_test_full.append(image.reshape(-1,128,128,3))
            angle = angles[j]
            angle = torch.tensor(angle, dtype=torch.float32)
            y_test_full.append(angle)

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)
    elif scenario == "age_torch":
        dataset_path = f"../SRGAN/imdb_wiki_data/imdb_preprocessed_128"
        with open(os.path.abspath(os.path.join(dataset_path, 'meta.json'))) as json_file:
            json_contents = json.load(json_file)
        image_names, ages = [], []
        for entry in json_contents:
            if isinstance(entry, dict):
                image_names.append(entry['image_name'])
                ages.append(entry['age'])
            else:
                image_name, age, gender = entry
                image_names.append(image_name)
                ages.append(age)
        image_names, ages = np.array(image_names), np.array(ages)
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        ages = min_max_scaler.fit_transform(ages.reshape(-1, 1)).reshape(-1)
        image_names = np.array(image_names)
        ages = np.array(ages, dtype=np.float32)
        length = int(image_names.shape[0]/2)
        full_length = int(image_names.shape[0])
        X_train = []
        y_train = []
        X_test_full = []
        y_test_full = []
        for i in range(0,5000):
            image_name = image_names[i]
            image = imageio.imread(os.path.join(dataset_path, image_name))
            image = image.transpose((2, 0, 1))
            image = torch.tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            age = ages[i]
            age = torch.tensor(age, dtype=torch.float32)
            X_train.append(image.reshape(-1,128,128,3))
            y_train.append(age)
        for j in range(5000,10000):
            image_name = image_names[j]
            image = imageio.imread(os.path.join(dataset_path, image_name))
            image = image.transpose((2, 0, 1))
            image = torch.tensor(image.astype(np.float32))
            image = (image / 127.5) - 1
            age = ages[j]
            age = torch.tensor(age, dtype=torch.float32)
            X_test_full.append(image.reshape(-1,128,128,3))
            y_test_full.append(age)

        X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)
    else:
        raise NotImplementedError("Dataset does not exist")

    return X_train, y_train, X_test, y_test, X_valid, y_valid
