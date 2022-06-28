"""
Code for the polynomial coefficient data generating models.
"""

import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import pickle
from tqdm import tqdm
import math
import random
import torch
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from scipy.stats import uniform
from torch.utils.data import Dataset
from utility import MixtureModel, seed_all

irrelevant_data_multiplier = 5


class ToyDataset(Dataset):
    def __init__(self, start=None, end=None, seed=None, batch_size=None):
        seed_all(seed)
        self.scenario = "pumadyn"
        #with open(os.path.join(self.dataset_path, 'data.txt')) as f:
        meta, scaling, size = get_dataset(scenario=self.scenario)
        #meta = pd.read_pickle(os.path.join(self.dataset_path, 'meta.pkl'))
        meta = sklearn.utils.shuffle(meta, random_state=seed)  # Shuffles only first axis.
        x = meta[:, 0:size]
        y = meta[:, size]*scaling
        #print(x)
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        y = min_max_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
        self.x = np.array(x[start:end])
        self.y = np.array(y[start:end], dtype=np.float32)
        # Force full batch sizes
        if self.x.shape[0] < batch_size:
            repeats = math.ceil(batch_size / self.x.shape[0])
            self.x = np.repeat(self.x, repeats, axis=0)
            self.y = np.repeat(self.y, repeats)
        self.length = self.y.shape[0]
        self.image_size = 128

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #image_name = self.image_names[index]
        #image = np.load(os.path.join(self.dataset_path, image_name.replace('.jpg', '.npy')))
        #image = torch.tensor(image.astype(np.float32))
        x = self.x[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = self.y[index]
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

def get_dataset(n_instance=1000, scenario="ailerons", seed=1):
    """
    Create regression data: y = x(1 + f(z)) + g(z)
    """

    if scenario == "ailerons":
        scaling = 10000

        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/Ailerons/ailerons.data", delimiter=',')
        size = 40
        #X_train = my_data_train[:, 0:40]
        #y_train = my_data_train[:, 40] * scaling

        #my_data_test = np.genfromtxt(f"../FuzzyGAN/data/Ailerons/ailerons.test", delimiter=',')
        #X_test_full = my_data_test[:, 0:40]
        #y_test_full = my_data_test[:, 40] * scaling

        #X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "comp-activ":
        scaling = 1

        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/comp-activ/Prototask.data", delimiter=' ')
        size = 21

        #X_train = my_data_train[0:4096, 0:21]
        #y_train = my_data_train[0:4096, 21] * scaling

        #X_test_full = my_data_train[4096:8192, 0:21]
        #y_test_full = my_data_train[4096:8192, 21] * scaling

        #X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "pumadyn":
        scaling = 1000
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/pumadyn-32nm/Prototask.data", delimiter=' ')
        size = 32
        from sklearn import preprocessing
        #min_max_scaler = preprocessing.MinMaxScaler()
        #my_data_train_scaled = min_max_scaler.fit_transform((my_data_train[0:8192, 32] * scaling).reshape(-1, 1))

        #X_train = my_data_train[0:4096, 0:32]
        #y_train = my_data_train_scaled[0:4096]

        #X_test_full = my_data_train[4096:8192, 0:32]
        #y_test_full = my_data_train_scaled[4096:8192]

        #X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "bank":
        scaling = 10
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/bank-32nm/Prototask.data", delimiter=' ')
        size = 32
        #X_train = my_data_train[0:4096, 0:32]
        #y_train = my_data_train[0:4096, 32] * scaling

        #X_test_full = my_data_train[4096:8192, 0:32]
        #y_test_full = my_data_train[4096:8192, 32] * scaling

        #X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "census-house":
        scaling = 10 ** -5
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/census-house/house-price-16H/Prototask.data",
                                      delimiter=' ')
        size = 16
        #X_train = my_data_train[0:11392, 0:16]
        #y_train = my_data_train[0:11392, 16] * scaling

        #X_test_full = my_data_train[11392:22784, 0:16]
        #y_test_full = my_data_train[11392:22784, 16] * scaling

        #X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)

    elif scenario == "abalone":
        scaling = 1
        my_data_train = np.genfromtxt(f"../FuzzyGAN/data/abalone/Prototask.data", delimiter=' ')
        size = 8
        #X_train = my_data_train[0:2089, 1:8]
        #y_train = my_data_train[0:2089, 8] * scaling

        #X_test_full = my_data_train[2089:4177, 1:8]
        #y_test_full = my_data_train[2089:4177, 8] * scaling

        #X_test, X_valid, y_test, y_valid = train_test_split(X_test_full, y_test_full, random_state=seed)


    else:
        raise NotImplementedError("Dataset does not exist")
    return my_data_train, scaling, size

