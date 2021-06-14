"""Contains method that generates the dataset."""
import torch
import numpy as np


def get_data(num_samples):
    """Generate a Gaussian semi-circle around the origin."""
    X_1 = np.random.uniform(low=-1.0, high=1.0, size=(num_samples,))
    X_2 = np.random.uniform(low=0.0, high=1.0, size=(num_samples,))
    X = np.c_[X_1, X_2]
    y = np.exp(-np.linalg.norm(X, 2, axis=1))[:, np.newaxis]
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    return X, y

