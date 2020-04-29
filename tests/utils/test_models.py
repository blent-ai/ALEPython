# -*- coding: utf-8 -*-
import numpy as np


class SimpleModel:
    """A simple predictive model for testing purposes.

    Methods
    -------
    predict(X)
        Given input data `X`, predict response variable.

    """

    def predict(self, X):
        return np.mean(X, axis=1)


def simple_predictor(X):
    return np.mean(X, axis=1)


def linear_predictor(X):
    """A simple linear effect with features 'a' and 'b'."""
    return X["a"] + X["b"]


def interaction_predictor(X):
    """Interaction changes sign at b = 0.5."""
    a = X["a"]
    b = X["b"]

    out = np.empty_like(a)

    mask = b <= 0.5
    out[mask] = a[mask] * b[mask]
    mask = ~mask
    out[mask] = -a[mask] * b[mask]

    return out
