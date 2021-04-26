from shaman import *
import argparse

def to_bool(text):
    if (text == "True" or text ==  "true"):
        return True
    return False


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='A typical linear regression')
    parser.add_argument("data_path", nargs='?', default="data.csv")
    parser.add_argument("standardize", nargs='?', default="True", type=to_bool)
    args = parser.parse_args()
    d = Dataset(args.data_path, standardize = args.standardize)
    shaman = Shaman(d, standardize=args.standardize)
    shaman.training_loop()
    shaman.write_thetas_to_file()
