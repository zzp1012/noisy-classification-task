import os
import numpy as np

class DataUtils:
    """Data utils class.
    """

    @classmethod
    def load(cls,
             data_path: str="../data", 
             train: bool=True):
        """Load the data from the data folder.

        Args:
            data_path (str, optional): Path to the data folder. Defaults to "../data".
            train (bool, optional): If True, load the training data. Defaults to True.
        
        Returns:
            features (np.ndarray): features.
            labels (np.ndarray): labels (None if train = False)
        """
        if train:
            # load the data 
            X = np.load(os.path.join(data_path, "train_features.npy"))
            y = np.load(os.path.join(data_path, "train_labels.npy"))

            # simple check
            assert len(X) == len(y), \
                "Number of features and labels do not match."
        else:
            # load the data 
            X = np.load(os.path.join(data_path, "test_features.npy"))
            y = None
        return X, y

    @classmethod
    def kfold_split(cls,
                    X: np.ndarray, 
                    y: np.ndarray, 
                    k = 5, 
                    seed = 0):
        """Split the data into k folds.

        Args:
            X (np.ndarray): features.
            y (np.ndarray): labels.
            k (int, optional): Number of folds. Defaults to 5.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
        
        Returns:
            splits (list): list of splits. contains X_train, y_train, X_val, y_val.
        """
        # simple check
        assert len(X) == len(y), \
            "Number of features and labels do not match."
        # shuffle the data
        np.random.seed(seed) # set the seed
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

        # split the data into k folds
        folds = []
        for i in range(k):
            folds.append((X[i::k], y[i::k]))

        splits = []
        for i in range(k):
            split = {}
            split["X_train"] = np.concatenate(
                [folds[j][0] for j in range(k) if j != i])
            split["y_train"] = np.concatenate(
                [folds[j][1] for j in range(k) if j != i])
            split["X_val"] = folds[i][0]
            split["y_val"] = folds[i][1]
            splits.append(split)
        return splits

    @classmethod
    def bootstrap_split(cls,
                        X: np.ndarray, 
                        y: np.ndarray, 
                        k: int = 5,
                        seed: int = 0):
        """Split the data into k folds using bootstrap.

        Args:
            X (np.ndarray): features.
            y (np.ndarray): labels.
            k (int, optional): Number of splits. Defaults to 5.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
        
        Returns:
            splits (list): list of splits. contains X_train, y_train, X_val, y_val.
        """
        # simple check
        assert len(X) == len(y), \
            "Number of features and labels do not match."
        # set the seed
        np.random.seed(seed)
        # split the data into k splits
        splits = []
        for i in range(k):
            split = {}
            sampled_indices = np.random.randint(0, len(X), len(X))
            split["X_train"] = X[sampled_indices]
            split["y_train"] = y[sampled_indices]
            split["X_val"] = X
            split["y_val"] = y
            splits.append(split)
        return splits


if __name__ == "__main__":
    # simple test
    # load the data
    X_train, y_train = DataUtils.load("../../data", train = True)
    print(X_train.shape, y_train.shape)

    # split the data into 5 splits
    splits = DataUtils.kfold_split(X_train, y_train, k = 5, seed = 0)
    print(len(splits))
    print(splits[0]["X_train"].shape, splits[0]["y_train"].shape)
    print(splits[0]["X_val"].shape, splits[0]["y_val"].shape)

    # split the data into 5 splits using bootstrap
    splits = DataUtils.bootstrap_split(X_train, y_train, k = 5, seed = 0)
    print(len(splits))
    print(splits[0]["X_train"].shape, splits[0]["y_train"].shape)
    print(splits[0]["X_val"].shape, splits[0]["y_val"].shape)