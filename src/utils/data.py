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
            train_features = np.load(os.path.join(data_path, "train_features.npy"))
            train_labels = np.load(os.path.join(data_path, "train_labels.npy"))

            # simple check
            assert len(train_features) == len(train_labels), \
                "Number of features and labels do not match."

            return train_features, train_labels
        else:
            # load the data 
            test_features = np.load(os.path.join(data_path, "test_features.npy"))
            return test_features, None

    @classmethod
    def kfold_split(cls,
                    features: np.ndarray, 
                    labels: np.ndarray, 
                    k = 5, 
                    seed = 0):
        """Split the data into k folds.

        Args:
            features (np.ndarray): features.
            labels (np.ndarray): labels.
            k (int, optional): Number of folds. Defaults to 5.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
        
        Returns:
            folds (list): list of folds.
        """
        # simple check
        assert len(features) == len(labels), \
            "Number of features and labels do not match."
        # shuffle the data
        np.random.seed(seed) # set the seed
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]

        # split the data into k folds
        folds = []
        for i in range(k):
            fold = {}
            fold["features"] = features[i::k]
            fold["labels"] = labels[i::k]
            folds.append(fold)
        return folds

    @classmethod
    def bootstrap_split(cls,
                        features: np.ndarray, 
                        labels: np.ndarray, 
                        k: int = 5,
                        seed: int = 0):
        """Split the data into k folds using bootstrap.

        Args:
            features (np.ndarray): features.
            labels (np.ndarray): labels.
            k (int, optional): Number of folds. Defaults to 5.
            seed (int, optional): Seed for the random number generator. Defaults to 0.
        
        Returns:
            folds (list): list of folds.
        """
        # simple check
        assert len(features) == len(labels), \
            "Number of features and labels do not match."
        # set the seed
        np.random.seed(seed)
        # split the data into k folds
        folds = []
        for i in range(k):
            fold = {}
            sampled_indices = np.random.randint(0, len(features), len(features))
            fold["features"] = features[sampled_indices]
            fold["labels"] = labels[sampled_indices]
            folds.append(fold)
        return folds


if __name__ == "__main__":
    # simple test
    # load the data
    features, labels = DataUtils.load("../../data", train = True)
    print(features.shape, labels.shape)

    # split the data into 5 folds
    folds = DataUtils.kfold_split(features, labels, k = 5, seed = 0)
    print(len(folds))
    print(folds[0]["features"].shape, folds[0]["labels"].shape)

    # split the data into 5 folds using bootstrap
    folds = DataUtils.bootstrap_split(features, labels, k = 5, seed = 0)
    print(len(folds))
    print(folds[0]["features"].shape, folds[0]["labels"].shape)