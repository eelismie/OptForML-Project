"""Contains method that generates the dataset."""
import torch
import numpy as np


def get_data(num_samples):
    """Generate a Gaussian semi-circle around the origin."""
    np.random.seed(0)

    W = np.array(
        [[1, 0],
         [1, 1]])

    X_1 = np.random.normal(size=(num_samples,))
    X_2 = np.absolute(np.random.normal(size=(num_samples,)))
    ang = np.argsort(np.arctan2(X_2, X_1)) #this sorts the samples by their angle
    X_old = np.c_[X_1, X_2][ang]
    X = X_old @ W

    y = np.exp(-(X_old**2).sum(axis=1)) - 1/4
    y = y[:, np.newaxis]

    # to torch tensor
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    return X, y
