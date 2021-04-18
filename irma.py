import pandas as pd

class Datapoint():
    def __init__(self, km, price):
        self.km = km
        self.price = price


    def __str__(self):
        return (f"km: {self.km}, price {self.price}")


    def __repr__(self):
        return (f"km: {self.km}, price {self.price}")

class Dataset():
    def __init__(self, path = None):
        self.data = None
        if (path != None):
            self.import_csv(path)


    def read_csv(self, path):
        self.data = pd.read_csv(path)


    def __getitem__(self, i):
        return Datapoint(self.data["km"][i], self.data["price"][i])
    

    def __len__(self):
        return (self.data.shape[0])

