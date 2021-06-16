"""Graph module."""
import math as m

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


class model_lr(nn.Module):
    """ Logistic regression w/out sigmoid output """

    def __init__(self, input_dim=15, output_dim=1):
        super(model_lr, self).__init__()
        self.l = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        out = self.l(x)
        return out


class model_nn(nn.Module):
    "Linear NN model"

    def __init__(self, n_layers=3, layer_size=5, hl_dim=10, in_dim=28 * 28, out_dim=10):
        super(model_nn, self).__init__()

        modules = []

        modules.extend([nn.Linear(in_dim, hl_dim), nn.ReLU()])
        for _ in range(n_layers):
            modules.extend([nn.Linear(hl_dim, hl_dim), nn.ReLU()])
        modules.append(nn.Linear(hl_dim, out_dim))

        self.network = nn.Sequential(*modules)


    def forward(self, x):
        out = self.network(x)

        return out

class node():
    """ node class to simulate training instance with separate dataset """

    def __init__(self, data_x, data_y, **kwargs):
        self.model = kwargs['model'](**kwargs['model_kwargs'])
        self.trainset = TensorDataset(data_x, data_y)
        self.train_generator = DataLoader(self.trainset, batch_size=kwargs['batch_size'])
        self.criteria = kwargs['criteria']()

    def parameters(self):
        return self.model.parameters()

    def forward_backward(self):
        for batch_x, batch_y in self.train_generator:
            out = self.model(batch_x)
            l = self.criteria(out, batch_y)
            l.backward()

class graph():
    """ Graph class to orchestrate training and combine weights from nodes """
    def __init__(self, data, W_matrix, iid=True, **kwargs):
        self.losses = []
        self.W_matrix = torch.from_numpy(W_matrix).to(torch.float32)
        # store global dataset for stats
        # tuple of features and targets
        self.data = data

        # non-iid data
        if not iid:
            self.process_non_iid()

        x_partitions, y_partitions = self.partition(pieces=self.W_matrix.shape[0])

        self.nodes = [node(x_partitions[i], y_partitions[i], **kwargs) for i in range(self.W_matrix.shape[0])]
        params = self.parameters()
        self.optim = kwargs['optimiser']([{'params' : p} for p in params], **kwargs['optimiser_kwargs'])

    def parameters(self):
        """Return all parameters from all nodes."""
        return [n.parameters() for n in self.nodes]

    def partition(self, pieces=1):
        """Partition preserving iid, assuming data is iid in the indices."""
        x = self.data[0]  # features
        y = self.data[1]  # targets

        x_partitions = []
        y_partitions = []

        rows = x.shape[0]
        size = m.floor(float(rows)/float(pieces))

        for i in range(pieces):
            x_partitions.append(x[i * size:(i + 1) * size, :])  # features
            y_partitions.append(y[i * size:(i + 1) * size])  # targets

        return x_partitions, y_partitions


    def process_non_iid(self):
        """Sort data by angle (only works with 2-D data)."""
        x = self.data[0]  # features
        y = self.data[1]  # targets

        # only 2-D samples
        if x.shape[1] != 2:
            return

        #TODO: non__iid_partitions
        #Maybe smarter just to study the effect that inexact averaging has on the stochastic convergence rates
        # this sorts the samples by their angle
        ang = np.argsort(np.arctan2(x[:, 1], x[:, 0]))
        self.data = (x[ang], y[ang])

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
                #this can be computed using the weight matrix and the numbe of parameters per model
                self.mix_weights()

            self.print_loss()

    def mix_weights(self):
        with torch.no_grad():
            N = len([_ for _ in self.nodes[0].model.parameters()])
            for i in range(N):
                params = []
                for n in self.nodes:
                    gen = n.model.parameters()
                    p = next(gen)
                    for _ in range(i):
                        p = next(gen)
                    params += [p]

                new_params = torch.stack(params)
                new_params = torch.tensordot(self.W_matrix, new_params, dims=([1], [0]))

                for i, p in enumerate(params):
                    p[:] = new_params[i]

    def print_loss(self):
        loss = 0.0
        nodes = self.W_matrix.shape[0]
        full_X = self.data[0]
        full_Y = self.data[1]

        for i in self.nodes:
            out = i.model(full_X)
            l = i.criteria(out, full_Y)
            loss += (1.0/nodes)*l.item()

        print('train loss :', loss)
        self.losses.append(loss)
