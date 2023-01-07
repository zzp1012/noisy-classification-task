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
            return SVC(probability=True, **kwargs)
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
    def cross_entropy(cls,
                      model: BaseEstimator,
                      X: np.ndarray,
                      y: np.ndarray,) -> float:
        """Calculate the cross entropy.
        
        Args:
            model (BaseEstimator): The model.
            X (np.ndarray): Features. (N, D)
            y (np.ndarray): Labels. (N, )
        
        Returns:
            loss (float): Cross entropy.
        """
        # simple check
        # simple check
        assert len(X.shape) == 2, \
            "X should be 2D."
        assert y.shape == (X.shape[0], ), \
            "y should be 1D and the length equals to X."

        y_prob = model.predict_proba(X) # (N, C)
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15) # avoid log(0)
        
        # calculate the loss
        loss = 0
        for i in range(y_prob.shape[0]):
            loss -= np.log(y_prob[i, y[i]])
        loss /= y_prob.shape[0]
        return loss


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
    # calculate the loss
    loss = ModelUtils.cross_entropy(model, X_train, y_train)
    print(loss)