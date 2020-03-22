'''
DreemDataset class
Loads the dataset and normalize it, balance it
and separates it if needed.

Usage:
dataset = DreemDataset("/content/drive/My Drive/Colab Notebooks/data/", balanced=True, separated=False)

Note:
This is not a PyTorch Dataset.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py

class DreemDataset:
    def __init__(self, DIR, normalize=True, balanced=True, separated=False, refactor=True):
        """
        Class Dataset, contains the train and test dataset
        and every function needed to split, normalize, etc.
        
        Args:
            DIR (string): directory where X_train.h5 and y_train.csv are located
            normalize (bool): True is the data had to be centered and reduced
            balanced (bool): True if the dataset has to be balanced
            separated (bool): True if the signals have to be split by patient
            refactor (bool): True if the input signals of each captors has to be agregated - not compatible with separated.
        """

        # Dataset
        self.X = np.array(h5py.File(DIR + "X_train.h5", "r")["features"])
        self.y = np.array(pd.read_csv(DIR + "y_train.csv")["label"])

        # Dimensions
        self.nb_data = self.X.shape[0]
        self.nb_receptors = self.X.shape[1]
        self.nb_sig_per_rec = self.X.shape[2]
        self.nb_signals = self.X.shape[1] * self.X.shape[2]
        self.signal_size = self.X.shape[3]

        # Train test variables
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        # Create balance
        if balanced:
            self.balance()
        
        # Normalize
        if normalize:
            self.normalize()

        # Train test split
        self.split()
        
        # Separate
        if separated:
            self.X_train, self.y_train = self.separate(self.X_train, self.y_train)
            self.X_test, self.y_test = self.separate(self.X_test, self.y_test)
        
        # Refactor
        if not separated and refactor:
            self.refactor()
    
    def balance(self):
        # Balance the dataset so that
        # there is 50% 50% class balance
        # The seed is set so that the results are reproducible

        np.random.seed(42)

        m0, m1 = self.y == 0, self.y == 1
        X0, X1 = self.X[m0], self.X[m1]
        y0, y1 = self.y[m0], self.y[m1]

        idx = np.random.randint(0, len(X0), len(X1))
        X0, y0 = X0[idx], y0[idx]

        X = np.concatenate((X0, X1))
        y = np.concatenate((y0, y1))

        # Shuffle
        idx = np.random.permutation(len(X))
        self.X, self.y = X[idx], y[idx]
    
    def separate(self, X, y):
        # Separate the signals coming from each patient, then shuffle
        # The seed is set for the shuffling

        new_X, new_Y = None, None
        for idx, (x, y) in enumerate(zip(X, y)):
            if new_X is not None:
                new_X = np.concatenate((new_X, x), axis=0)
                new_Y = np.concatenate((new_Y, [y]*len(x)), axis=0)
            else:
                new_X = x
                new_Y = np.array([y] * len(x))
        
        # Shuffle
        np.random.seed(43)
        idx = np.random.permutation(len(new_X))

        return new_X[idx], new_Y[idx]
    
    def refactor(self):
        # Reshape the input signal so that it is
        # of shape [nb_data, nb_signals, signal_size]

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.nb_signals, self.signal_size)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.nb_signals, self.signal_size)
    
    def normalize(self):
        # Normalize and reduce the dataset
        # so that each signal of size 500 has mean 0 and std 1, globally

        mean = self.X.mean(axis=(0,1,3), keepdims=True)
        std = self.X.std(axis=(0,1,3), keepdims=True)

        self.X = (self.X - mean) / std
    
    def split(self):
        # Train test split

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, shuffle=True, random_state=42)
    
    def iter_train(self, batch_size, shuffle=True):
        n = len(self.X_train)

        for i in range(0, n, batch_size):
            idx = np.random.randint(0, n, batch_size)
            x, y = self.X_train[idx], self.y_train[idx]

            # data augmentation
            if shuffle:
                # [40*7, 500]
                x = x.reshape((batch_size, 40, 7, 500))
                new_x = np.zeros(x.shape)
                for k in range(batch_size):
                    for l in range(7):
                        permut = np.random.permutation(40)
                        new_x[k, :, :, :] = x[k, permut, :, :]
                
                x = new_x.reshape((batch_size, 40*7, 500))

            yield x, y

    def iter_test(self, batch_size):
        n = len(self.X_test)

        for i in range(0, n, batch_size):
            x, y = self.X_test[i:i+batch_size], self.y_test[i:i+batch_size]

            yield x, y
