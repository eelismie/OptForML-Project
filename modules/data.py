"""Contains method that generates the dataset."""
import torch
import numpy as np


def get_data(num_samples):

    """Generate a Gaussian semi-circle around the origin."""
    np.random.seed(0)
    X_1 = np.random.normal(size=(num_samples,))
    X_2 = np.absolute(np.random.normal(size=(num_samples,)))
    ang = np.argsort(np.arctan(X_2 / X_1)) #this sorts the samples by their angle 
    X = np.c_[X_1, X_2][ang] 
    y = np.exp(-np.linalg.norm(X, 2, axis=1))[:, np.newaxis]
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    return X, y
