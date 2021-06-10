import math as m

import torch
import torch.nn as nn
import torchvision
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from IPython import embed


class model_lr(nn.Module):
    """ Logistic regression w/out sigmoid output """

    def __init__(self, input_dim=15, output_dim=1):
        super(model_lr, self).__init__()
        self.l = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        out = self.l(x)
        return out


class node():
    """ node class to simulate training instance with separate dataset """

    def __init__(self, data_x, data_y, **kwargs):
        self.model = kwargs['model'](**kwargs['model_kwargs'])
        # self.X = data_x
        # self.Y = data_y
        self.trainset = TensorDataset(data_x, data_y)
        self.train_generator = DataLoader(self.trainset, batch_size=kwargs['batch_size'])
        self.criteria = kwargs['criteria']()

    def parameters(self):
        return self.model.parameters()

    def forward_backward(self):
        #TODO: make compatible with local minibatch updates as well
        for batch_x, batch_y in self.train_generator:
            out = self.model(batch_x)
            l = self.criteria(out, batch_y)
            l.backward()

class graph():
    """ graph class to orchestrate training and combine weights from nodes """
    def __init__(self, data, W_matrix, **kwargs):

        self.W_matrix = torch.from_numpy(W_matrix).to(torch.float32)
        x_partitions, y_partitions = self.partition(data, pieces=self.W_matrix.shape[0])
        self.nodes = [node(x_partitions[i], y_partitions[i], **kwargs)  for i in range(self.W_matrix.shape[0])]
        params = self.parameters()
        self.optim = kwargs['optimiser']([{'params' : p} for p in params], **kwargs['optimiser_kwargs'])

    def parameters(self):
        """Return all parameters from all nodes."""
        return [n.parameters() for n in self.nodes]

    def partition(self, data, pieces=1):
        x = data[0]
        y = data[1]

        x_partitions = []
        y_partitions = []

        rows = x.shape[0]
        size = m.floor(float(rows)/float(pieces))

        for i in range(pieces):
            x_partitions.append(x[i*size:(i + 1)*size,:]) #features
            y_partitions.append(y[i*size:(i + 1)*size]) #targets

        return x_partitions, y_partitions


    def non_iid_partition(self):
        raise NotImplementedError


    def run(self, mixing_steps=1, local_steps=1, iters=100):

        for iter_ in range(iters):

            #run training in each node

            for local_ in range(local_steps):
                for node in self.nodes:
                    node.forward_backward()

                self.optim.step()
                self.optim.zero_grad()

            for mix_ in range(mixing_steps):
                #TODO: track number of communications with other nodes. would be interesting to look into total communication costs
                self.mix_weights()

            self.print_loss()

    def mix_weights(self):
        #TODO : implement correct weight mixing
        #need to first store data on parameters,
        #then they need to be mixed correctly and placed in the right models
        with torch.no_grad():
            params = [n.model.l.weight for n in self.nodes]

            params_tensor = torch.stack(params)
            new_params = torch.tensordot(self.W_matrix, params_tensor, dims=([1], [0]))

            for i, p in enumerate(new_params):
                # print('change 1: ', torch.norm(p - params[i]).item())
                params[i][:] = p
                # print('change 2: ', torch.norm(p - params[i]).item())

    def print_loss(self):
        node = self.nodes[0]
        X, y = node.trainset[:]
        out = node.model(X)
        l = node.criteria(out, y)
        print(l.item())


def ring_topo(num_elems):
    result = np.zeros((num_elems, num_elems))
    for i in range(num_elems):
        result[i, (i + 1)%num_elems] = 1/3
        result[i, (i + num_elems - 1)%num_elems] = 1/3
        result[i,i] = 1/3
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
    """Metropolis Hastings weight assignment for distributed averaging"""
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


if __name__=="__main__":
    model_kwargs = {"input_dim" : 784, "output_dim" : 2}
    optimiser_kwargs = {"lr" : 0.001} #specify keyword args for model

    graph_kwargs = {"model_kwargs": model_kwargs, #pass model kwargs
        "optimiser_kwargs" : optimiser_kwargs,
        "criteria" : nn.CrossEntropyLoss, #specify loss function for each node
        "model" : model_lr, #specify model class handle
        "optimiser" : torch.optim.SGD, #specify global optimiser
        "batch_size" : 100 #specify batch size for each node
        }

    #load data
    samples= 2000
    tot_samples = samples*2
    dataset = torchvision.datasets.MNIST(root = "data",train=True,download=True )
    idx_0 = (dataset.train_labels==0)
    idx_1 = (dataset.train_labels==1)

    perm = torch.randperm(samples)
    x = torch.cat([dataset.train_data[idx_0][:samples], dataset.train_data[idx_1][:samples]]) \
            .reshape(tot_samples, -1) \
            .type(torch.FloatTensor)[perm,:]
    y = torch.cat([dataset.train_labels[idx_0][:samples], dataset.train_labels[idx_1][:samples]]) \
            .type(torch.LongTensor)[perm]

    data = (x, y)
    W_matrix = fc_topo(5) #define fully connected topology with 4  nodes

    graph_1 = graph(data, W_matrix, **graph_kwargs)
    graph_1.run(mixing_steps=1, local_steps=1, iters=10) #this should be equivalent to a single training step in the non_distributed case
