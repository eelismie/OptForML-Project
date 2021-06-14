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

        perm = torch.randperm(data[0].shape[0])

        x = data[0][perm] #shuffle data
        y = data[1][perm]

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
                self.mix_weights()

            #self.print_loss()
            self.write_train_loss()

    def mix_weights(self):
        with torch.no_grad():
            weights = [n.model.l.weight for n in self.nodes]
            biases = [n.model.l.bias for n in self.nodes]

            weights = torch.stack(weights)
            biases = torch.stack(biases)

            weights = torch.tensordot(self.W_matrix, weights, dims=([1], [0]))
            biases = torch.tensordot(self.W_matrix, biases, dims=([1], [0]))

            for i, n in enumerate(self.nodes):
                # print('change 1: ', torch.norm(n.model.l.bias - biases[i]).item())
                n.model.l.weight[:] = weights[i]
                n.model.l.bias[:] = biases[i]
                # print('change 1: ', torch.norm(n.model.l.bias - biases[i]).item())

    def print_loss(self):
        node = self.nodes[0]
        X, y = node.trainset[:]
        out = node.model(X)
        l = node.criteria(out, y)
        print(l.item())

    def write_train_loss(self):

        """ total loss across nodes assuming equal size partitions of data """

        loss = 0.0 
        nodes = self.W_matrix.shape[0]

        for i in self.nodes:
            X, Y = i.trainset[:]
            out = i.model(X)
            l = i.criteria(out, Y)
            loss += (1.0/nodes)*l.item()
        
        self.losses.append(loss)
