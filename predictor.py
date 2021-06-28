import argparse
from datetime import datetime
from dataset import *

float_formatter = "{:.2E}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

class BabyShaman():
    def __init__(self, thetas_path = "thetas.csv"):
        try:
            self.thetas = np.genfromtxt(thetas_path, delimiter=',')
        except Exception as e:
            print(f"There was an issue reading the thetas:\n{e}")
            raise ValueError


    def predict(self, data, thetas = None):
        if thetas is None:
            thetas = self.thetas
        return (np.dot(data, thetas))

    def __str__(self):
        return (f"Thetas: {self.thetas}")



def read_thetas(self, path = 'thetas.csv'):
    self.thetas = np.genfromtxt(path, delimiter=',')

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='A typical linear regression')
    shaman = BabyShaman("thetas.csv")
    x = input("enter the km's of the car\n")
    x = [int(x)]
    if (len(x) + 1 != len(shaman.thetas)):
        print("Your thetas and your input are of incompatible shape.")
        raise ValueError
    x = np.concatenate(([1], x))
    print("The predicted price is: ", shaman.predict(x))
