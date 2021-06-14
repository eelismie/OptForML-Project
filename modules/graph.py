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
        # self.X = data_x
        # self.Y = data_y
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
        """ Partitioning car data for nodes in a non-iid fashion """
        test_data = data[1].drop("name", axis=1)
        x_test = torch.from_numpy(np.array(test_data.drop("selling_price", axis=1))).float()
        y_test = torch.from_numpy(np.array(test_data["selling_price"])).float()

        self.test_data = (x_test, y_test)

        train_data = data[0]
        # sort by car brand so each node is assigned mostly one brand (or whatever else if we use another dataset)
        train_data.sort_values("name", inplace=True)
        x_train = torch.from_numpy(np.array(train_data.drop(["selling_price", "name"], axis=1))).float()
        y_train = torch.from_numpy(np.array(train_data["selling_price"])).float()

        mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
        x_train.sub_(mu).div_(sd)

        x_partitions = []
        y_partitions = []

        size = m.floor(float(x_train.shape[0]) / float(pieces))

        for i in range(pieces):
            x_partitions.append(x_train[i * size: (i + 1) * size].view(-1, x_train.shape[1])) #features
            y_partitions.append(y_train[i * size: (i + 1) * size].unsqueeze(-1)) #targets

        return x_partitions, y_partitions

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
                    # print('change 1: ', torch.norm(p - new_params[i]).item())
                    p[:] = new_params[i]
                    # print('change 2: ', torch.norm(p - new_params[i]).item())

    def print_loss(self):
        node = self.nodes[0]
        X, y = node.trainset[:]
        out = node.model(X)
        l = node.criteria(out, y)
        print(l.item())
