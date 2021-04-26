import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler 


class Datapoint():
    def __init__(self, km, price):
        self.km = km
        self.price = price


    def __str__(self):
        return (f"km: {self.km}, price {self.price}")


    def __repr__(self):
        return (f"km: {self.km}, price {self.price}")

class Dataset():
    def __init__(self, path, standardize = True):
        self.data = None
        self.i = -1
        try:
            self.read_csv(path)
        except Exception as e:
            print(f"Please give a valid input, only numeric data is accepted\n{e}")
            raise ValueError
        self.standardized = False
        if (self.standardize):
            self.standardize()
        self.add_ones_to_x()


    def standardize(self):
        self.standardized = True
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.y = self.y[:, np.newaxis]
        self.x_scaler.fit(self.x)
        self.y_scaler.fit(self.y)
        self.x = self.x_scaler.transform(self.x)
        self.y = self.y_scaler.transform(self.y)
        self.y = np.reshape(self.y, self.y.shape[0])

    def read_csv(self, path):
        self.data = pd.read_csv(path, dtype = np.float64).to_numpy()
        try:
            self.p = self.data.shape[1] - 1
            self.m = self.data.shape[0]
        except:
            logging.error(f"Input needs to have at least two dimensions. Input dimension was {self.data.shape}")
            raise ValueError

        self.x = self.data[:, [x for x in range(self.p)]]
        self.y = self.data[:, self.p]

    def add_ones_to_x(self):
        self.x = np.concatenate((np.ones([self.m, 1], dtype = self.x.dtype), self.x), axis = 1)



    def __getitem__(self, i):
        return (self.x[i], self.y[i])
    

    def __len__(self):
        return (self.data.shape[0])

    
    def __iter__(self):
        self.i = -1
        return (self)

    def __next__(self):
        self.i += 1
        if (self.i < len(self)):
            return self[self.i]
        else:
            self.i = -1
            raise StopIteration