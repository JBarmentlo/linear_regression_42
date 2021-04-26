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
    parser.add_argument("x", nargs='+', type=float)
    parser.add_argument("thetas_path", nargs='?', default="thetas.csv")
    args = parser.parse_args()
    shaman = BabyShaman(args.thetas_path)
    if (len(args.x) + 1 != len(shaman.thetas)):
        print("Your thetas and your input are of incompatible shape.")
        raise ValueError
    x = np.concatenate(([1], args.x))
    print(shaman.predict(x))
