"""Method utilities."""
import numpy as np
from pathlib import Path


def open_csv(file_name, header):
    """Create or open csv file, and add header `header` in the former case."""
    # check if file exists
    if Path(file_name).is_file():
        # append to file and don't add header
        csv = open(file_name, 'a')
    else:
        # create file
        csv = open(file_name, 'w')
        # header
        csv.write(header + '\n')
    return csv


def get_bandwidth(W, params, mix_steps):
    """Return bytes per iteration for the given distributed system parameters."""
    # count connections. Subtract nodes since we don't count themselves
    nb_conn = (np.abs(W) > 1e-6).sum() - W.shape[0]
    # we use float32 == 4 bytes
    B = 4
    mem = B * sum([p.numel() for p in params])
    return nb_conn * mem * mix_steps

def get_beta(W):
    """Return theoretical variable which relates the neighborhood of convergence for a topology."""
    eigs = np.linalg.eig(W)[0]
    # ascending order
    eigs.sort()
    beta = max(np.abs(eigs[-2]), np.abs(eigs[0]))
    return beta