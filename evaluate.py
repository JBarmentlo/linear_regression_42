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
    try:
        parser = argparse.ArgumentParser(description='A typical linear regression')
        shaman = BabyShaman("thetas.csv")
        d = Dataset("data/data.csv", standardize = False)
        y = shaman.predict(d.x)
        error = np.abs(d.y - y)
        relative_error = error / d.y
        print(f"The average absolute error is {np.mean(error):.1f}.\nThe average relative error is {np.mean(relative_error) * 100:.2f}%.")
    except:
        print("This program was written to help understand the importance and impact of different settings, not be a robust implementation, use with caution and with the subject dataset.")