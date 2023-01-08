import os, re
import argparse
import pandas as pd

# import internal libs
from utils.avgmeter import MetricTracker

def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="train deep nn.")
    ## the basic setting of exp
    parser.add_argument("--data_path", default=None, type=str,
                        help='the path of loading data.')
    parser.add_argument("--method", default="bootstrap", type=str,
                        help='the evaluation method.')
    args = parser.parse_args()
    return args


def main():
    # get the args.
    args = add_args()

    # get the model name
    if "knn" in args.data_path:
        model_name = "knn"
    elif "mlp" in args.data_path:
        model_name = "mlp"
    elif "svm" in args.data_path:
        model_name = "svm"
    else:
        raise NotImplementedError

    # find all the files recursively in the data_path
    data_dict = {} 
    for root, _, files in os.walk(args.data_path):
        for file in files:
            if file == "loss.csv" and args.method in root:
                if model_name == "knn":
                    start, end = re.search(r"n_neighbors\d+", root).span()
                    key = root[start+11:end]
                elif model_name == "mlp":
                    start, end = re.search(r"\d+\+\d+", root).span()
                    key = root[start:end]
                elif model_name == "svm":
                    start, end = re.search(r"svm/[a-z]+", root).span()
                    key = root[start+4:end]
                else:
                    raise NotImplementedError
                
                # read the data
                data = pd.read_csv(os.path.join(root, file))
                loss = data["loss"].values.mean()
                data_dict[key] = loss
    
    # sort the data_dict
    data_dict = dict(sorted(data_dict.items(), 
        key=lambda item: int(item[0].split("+")[0]) if item[0][0].isdigit() else item[0]))

    # plot bar using pandas
    if model_name == "knn":
        xaxis = "n_neighbors"
    elif model_name == "mlp":
        xaxis = "hidden_layers"
    elif model_name == "svm":
        xaxis = "kernel"
    else:
        raise NotImplementedError
    df = pd.DataFrame(data_dict.items(), columns=[xaxis, "loss"])

    # plot the bar
    ax = df.plot.bar(x=xaxis, y="loss", rot=0)
    ax.set_xlabel(xaxis)
    ax.set_ylabel("loss")
    ax.set_title(f"{args.method} loss")
    ax.figure.savefig(os.path.join(args.data_path, f"{model_name}_{args.method}_loss.png"))

if __name__ == "__main__":
    main()