"""Contains method that generates the dataset."""
import torch
import numpy as np


def get_data(num_samples):
    """Generate a Gaussian semi-circle around the origin."""
    np.random.seed(0)

    W = np.array(
        [[1, 0],
         [0, 1]])

    X_1 = np.random.normal(size=(num_samples,))
    X_2 = np.absolute(np.random.normal(size=(num_samples,)))

    X_old = np.c_[X_1, X_2]
    X = X_old @ W

    y = np.exp(-(X_old**2).sum(axis=1)) - 1/4
    y = y[:, np.newaxis]

    # convert to torch tensors
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    return X, y
