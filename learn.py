from shaman import *
import argparse

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='A typical linear regression')
    parser.add_argument("data_path", nargs='?', default="data/data.csv")
    parser.add_argument("--plot", help="plot curves while training", action="store_true")
    args = parser.parse_args()
    d = Dataset(args.data_path, standardize = True)
    shaman = Shaman(d, standardize = True, plot = args.plot, time_limit = 2.0)
    shaman.training_loop()
    shaman.write_thetas_to_file()
    shaman.dataset.destandardize()
    if args.plot:
        shaman.graph.plot_final_result(shaman)
