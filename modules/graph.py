"""Graph module."""
import math as m

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


class model_lr(nn.Module):
    """ Linear model """

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
        self.data = data #store global dataset for stats

        if iid:
            x_partitions, y_partitions = self.partition(data, pieces=self.W_matrix.shape[0])
        else:
            x_partitions, y_partitions = self.non_iid_partition(data, pieces=self.W_matrix.shape[0])

        self.nodes = [node(x_partitions[i], y_partitions[i], **kwargs) for i in range(self.W_matrix.shape[0])]
        params = self.parameters()
        self.optim = kwargs['optimiser']([{'params' : p} for p in params], **kwargs['optimiser_kwargs'])

    def parameters(self):
        """Return all parameters from all nodes."""
        return [n.parameters() for n in self.nodes]

    def partition(self, data, pieces=1):

        """ data = tuple of features and labels """

        x = data[0]
        y = data[1]

        x_partitions = []
        y_partitions = []

        rows = x.shape[0]
        size = m.floor(float(rows)/float(pieces))

        for i in range(pieces):
            x_partitions.append(x[i * size:(i + 1) * size, :]) #features
            y_partitions.append(y[i * size:(i + 1) * size]) #targets

        return x_partitions, y_partitions

    def non_iid_partition(self, data, pieces=1):

        """ partittion data in non-iid way (assume preprocessed data)

        x = torch tensor with features
        y = torch tensor with labels

        """

        x = data[0]   #features
        y = data[1]   #classes

        #TODO: non__iid_partitions
        #Maybe smarter just to study the effect that inexact averaging has on the stochastic convergence rates
        pass

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

        print('train loss :', loss - 0.05307472)
        self.losses.append(loss - 0.05307472)

    def compute_communication(self, mixing_steps):
        nod = self.nodes[0]
        pp = 0
        for p in nod.model.parameters():
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn

        connections = np.count_nonzero(self.W_matrix)

        comms = connections*pp*mixing_steps
        return comms
