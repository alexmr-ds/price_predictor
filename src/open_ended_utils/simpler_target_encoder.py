import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    """
    A simple target mean encoder for categorical features.

    Each categorical value is replaced with the mean of the target variable
    observed for that category in the training data. Categories unseen during
    fitting are encoded using the global target mean.

    This encoder:
    - Operates independently per feature
    - Does not apply smoothing or regularization
    - Does not perform cross-validation
    - Is clone-safe and compatible with scikit-learn Pipelines

    Warning
    -------
    This encoder uses the target directly and can introduce target leakage
    if applied outside a proper training-only context. It should only be
    used within a Pipeline and evaluated using cross-validation.
    """

    def __init__(self,):
        pass

    def fit(self, X, y):
        """
        Fit the encoder by computing target means per category.

        For each categorical feature, the mean of the target variable is
        computed for every observed category. The global target mean is also
        stored for use with unseen categories.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Input feature matrix containing categorical features.

        y : array-like
            Target variable corresponding to `X`.

        Returns
        -------
        self : MeanTargetEncoder
            Fitted encoder with learned category-to-mean mappings.
        """
        X = pd.DataFrame(X, copy=True)
        y = pd.Series(y)

        self.global_mean_ = y.mean()
        self.cols = X.columns.tolist()
        self.mapping_ = {}

        for col in self.cols:
            self.mapping_[col] = y.groupby(X[col]).mean().to_dict()

        return self

    def transform(self, X):
        """
        Transform categorical features into their target mean encodings.

        Each category is replaced with its learned target mean. Categories
        that were not observed during fitting are encoded using the global
        target mean.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Input feature matrix containing categorical features.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, n_features) containing the encoded
            numeric representations of the input categorical features.
        """
        X = pd.DataFrame(X, copy=True)

        X_out = pd.DataFrame(index=X.index)

        for col in self.cols:
            mapping = self.mapping_[col]
            X_out[col] = X[col].map(mapping).fillna(self.global_mean_)

        return X_out.values