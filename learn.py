from shaman import *
import argparse


if __name__  == "__main__":
    parser = argparse.ArgumentParser(description='A typical linear regression')
    parser.add_argument("data_path", nargs='?', default="data/data.csv")
    parser.add_argument("--time", default=2, help="time limit for learning (seconds), defaults to 2.", action = 'store', nargs = "?", type=int)
    parser.add_argument("--no_standardize", help='do not standardize training data', action='store_false')
    parser.add_argument("--plot", help="plot curves while training", action="store_true")
    args = parser.parse_args()
    d = Dataset(args.data_path, standardize = args.no_standardize)
    shaman = Shaman(d, standardize = args.no_standardize, plot = args.plot, time_limit = args.time)
    shaman.training_loop()
    shaman.write_thetas_to_file()
    shaman.dataset.destandardize()
    if args.plot:
        shaman.graph.plot_final_result(shaman)
