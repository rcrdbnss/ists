import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRMasker(BaseEstimator, TransformerMixin):

    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.lbound, self.ubound = 0., 0.

    def fit(self, X, y=None):
        Q1 = np.nanquantile(X, 0.25)
        Q3 = np.nanquantile(X, 0.75)
        IQR = Q3 - Q1

        # Define the acceptable range
        self.lbound = Q1 - self.multiplier * IQR
        self.ubound = Q3 + self.multiplier * IQR
        return self

    def transform(self, X):
        X = X.copy()
        X[(X < self.lbound) | (X > self.ubound)] = None
        return X
