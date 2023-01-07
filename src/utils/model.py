import numpy as np
from sklearn.base import BaseEstimator

class ModelUtils:
    """Class for model utils. Mainly sklearn models."""
    
    @classmethod
    def auto(cls,
             model_name: str,
             **kwargs) -> BaseEstimator:
        """Auto load the model. Using scikit-learn Package.
        
        Args:
            model_name (str): Name of the model.
        
        Returns:
            model (nn.Module): The model.
        """
        if model_name == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**kwargs)
        elif model_name == "svm":
            from sklearn.svm import SVC
            return SVC(**kwargs)
        elif model_name == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**kwargs)
        else:
            raise ValueError("model_name not found.")

    @classmethod
    def train(cls,
              model: BaseEstimator,
              X: np.ndarray,
              y: np.ndarray,) -> BaseEstimator:
        """Train the model.
        
        Args:
            model (BaseEstimator): The model.
            X (np.ndarray): Features. (N, D)
            y (np.ndarray): Labels. (N, )
        
        Returns:
            model (BaseEstimator): The trained model.
        """
        # simple check
        assert len(X.shape) == 2, \
            "X should be 2D."
        assert y.shape == (X.shape[0], ), \
            "y should be 1D and the length equals to X."

        model.fit(X, y)
        return model

    @classmethod
    def predict(cls,
                model: BaseEstimator,
                X: np.ndarray) -> np.ndarray:
        """Predict the labels.
        
        Args:
            model (BaseEstimator): The model.
            X (np.ndarray): Features. (N, D)
        
        Returns:
            y_pred (np.ndarray): Predicted labels. (N, )
        """
        # simple check
        assert len(X.shape) == 2, \
            "X should be 2D."

        y_pred = model.predict(X)
        return y_pred

    @classmethod
    def predict_proba(cls,
                      model: BaseEstimator,
                      X: np.ndarray) -> np.ndarray:
        """Predict the probabilities of labels.
        
        Args:
            model (BaseEstimator): The model.
            X (np.ndarray): Features. (N, D)
        
        Returns:
            y_pred (np.ndarray): Predicted labels. (N, )
        """
        # simple check
        assert len(X.shape) == 2, \
            "X should be 2D."

        y_pred = model.predict_proba(X)
        return y_pred

    @classmethod
    def precision(cls,
                  y_true: np.ndarray,
                  y_pred: np.ndarray):
        """Calculate the precision.
        
        Args:
            y_true (np.ndarray): True labels. (N, )
            y_pred (np.ndarray): Predicted labels. (N, )
        
        Returns:
            precision (float): The precision.
        """
        # simple check
        assert y_true.shape == y_pred.shape == (y_true.shape[0], ), \
            "y_true and y_pred should have the same shape."

        precision = np.sum(y_true == y_pred) / y_true.shape[0]
        return precision


if __name__ == "__main__":
    # simple test
    from data import DataUtils
    # load the data
    X_train, y_train = DataUtils.load("../../data", train = True)
    print(X_train.shape, y_train.shape)
    print(y_train[:10])
    # train the model
    model = ModelUtils.auto("mlp", hidden_layer_sizes = (100, 100))
    model = ModelUtils.train(model, X_train, y_train)
    # predict
    y_pred = ModelUtils.predict(model, X_train)
    print(y_pred.shape)
    print(y_pred[:10])
    # precict proba
    y_pred_proba = ModelUtils.predict_proba(model, X_train)
    print(y_pred_proba.shape)
    print(y_pred_proba[0, :10])
    # precision
    precision = ModelUtils.precision(y_train, y_pred)
    print(precision)