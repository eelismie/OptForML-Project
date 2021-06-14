"""Contains method to generate different topology matrices."""
import numpy as np


def ring_topo(num_elems):
    result = np.zeros((num_elems, num_elems))
    for i in range(num_elems):
        result[i, (i + 1) % num_elems] = 1 / 3
        result[i, (i + num_elems - 1) % num_elems] = 1 / 3
        result[i,i] = 1 / 3
    return result


def fc_topo(num_elems):
    result = np.ones((num_elems, num_elems))
    result = result/num_elems
    return result


def random_topo(num_elems): # might be interesting to consider other random graph generating techniques
    """Create random symmetric topology"""
    result = np.random.randint(0, 2, size=(num_elems, num_elems))
    np.fill_diagonal(result, 1.0)
    result = result + result.T

    result[result <= 1] = 0
    result[result.nonzero()] = 1.0

    # ensure connectedness
    while not all([ele >= 2 for ele in result.sum(axis=1)]) == True:
        result = random_topo(num_elems)

    np.testing.assert_array_equal(result, result.T)

    return result.astype(float)


def MH_weights(w):
    """Metropolis Hastings weight assignment for distributed averaging."""
    degrees = w.sum(axis=1) - 1 # -1 to subtract itself as a neigbor

    result = np.zeros(w.shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if i != j and w[i, j] != 0:
                result[i, j] = 1.0 / max(degrees[i], degrees[j])
        result[i, i] = 1.0 - result[i].sum()

    # check for symmetry and stochasticity
    np.testing.assert_allclose(result.sum(axis=0), np.ones(w.shape[0]))
    np.testing.assert_allclose(result.sum(axis=1), np.ones(w.shape[0]))
    np.testing.assert_array_equal(result, result.T)

    return result
