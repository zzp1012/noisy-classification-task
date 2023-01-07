import os
import argparse
import pandas as pd

# import internal libs
from utils.data import DataUtils
from utils.model import ModelUtils
from utils.avgmeter import MetricTracker
from utils import get_datetime, set_logger, \
    get_logger, log_settings

def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="train deep nn.")
    ## the basic setting of exp
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--model", default="mlp", type=str,
                        help='the model name.')
    parser.add_argument("--method", default="bootstrap", type=str,
                        help='the evaluation method.')
    parser.add_argument("-k", "--n_splits", default=5, type=int,
                        help='the number of splits.')
    parser.add_argument("-p", "--predict", action="store_true",
                        help='if True, predict the test data.')
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.model}",
                         f"{args.method}",
                         f"n_splits{args.n_splits}",])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args

def main():
    # get the args.
    args = add_args()
    
    # set the logger
    set_logger(args.save_path)

    # get the logger
    logger = get_logger(f"{__name__}.main")

    # show the args.
    logger.info("#########parameters settings....")
    log_settings(args)

    # get the data
    logger.info("#########loading data....")
    X_train, y_train = DataUtils.load(train=True)
    logger.info(f"""X_train.shape: {X_train.shape}
                    y_train.shape: {y_train.shape}""")

    # using bootstrap or cross validation
    if args.method == "bootstrap":
        splits = DataUtils.bootstrap_split(
            X_train, y_train, k=args.n_splits, seed=args.seed)
    elif args.method == "kfold":
        splits = DataUtils.kfold_split(
            X_train, y_train, k=args.n_splits, seed=args.seed)
    else:
        raise NotImplementedError(
            f"not implemented {args.method}.")

    # train the model
    logger.info("#########training and eval model....")
    tracker = MetricTracker()
    for i, split in enumerate(splits):
        X_train_split, y_train_split, X_val, y_val = \
            split["X_train"], split["y_train"], split["X_val"], split["y_val"]

        # init the model
        model = ModelUtils.auto(args.model)
        # train the model
        model = ModelUtils.train(model, X_train_split, y_train_split)

        # evaluate the model
        val_loss = ModelUtils.cross_entropy(model, X_val, y_val)
        logger.info(f"loss: {val_loss} for split {i}.")

        # update the tracker
        tracker.track({
            "loss": val_loss,
            "split": i,
        })
    
    # save the tracker
    tracker.save_to_csv(os.path.join(args.save_path, "loss.csv"))

    # predict the test data
    if args.predict:
        logger.info("#########predicting test data....")
        # init the model
        model = ModelUtils.auto(args.model)
        # train the model
        model = ModelUtils.train(model, X_train, y_train)

        X_test, _ = DataUtils.load(train=False)
        y_pred = ModelUtils.predict(model, X_test)

        # save the prediction
        pred_dict = {
            "Id": list(range(len(y_pred))),
            "Category": y_pred.tolist(),
        }
        pd.DataFrame(pred_dict).to_csv(
            os.path.join(args.save_path, "pred.csv"), index=False)


if __name__ == "__main__":
    main()