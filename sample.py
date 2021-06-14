import math as m
from IPython import embed

import torch
import pandas as pd
import torch.nn as nn
import torchvision
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import networkx as nx #for graph visualization 

from utils import preprocess_car_data, car_train_test, car_to_torch, synthetic_data


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
    

def ring_topo(num_elems):
    result = np.zeros((num_elems, num_elems))
    for i in range(num_elems):
        result[i, (i + 1) % num_elems] = 1 / 3
        result[i, (i + num_elems - 1) % num_elems] = 1 / 3
        result[i,i] = 1 / 3
    return result

def draw_graph(w_matrix):
    """ use networkx to visualize the graph topology """

    G = nx.Graph()
    copy = np.copy(w_matrix)
    for i in range(w_matrix.shape[0]):
        copy[i,i] = 0
        row = copy[i,:]
        nonzero = np.nonzero(row) 
        for j in nonzero[0]:
            if (j > i):
                G.add_edge(i + 1, j + 1)
        
    nx.draw(G)
    plt.show()

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

    np.random.seed(3)

    df_car = pd.read_csv("data/cars.csv")
    df_car = preprocess_car_data(df_car)
    df_train, df_test = car_train_test(df_car)
    #x_train, y_train, x_test, y_test = car_to_torch(df_train, df_test)
    x_train, y_train, w = synthetic_data(1000, 7)
    data = (x_train, y_train)

    W_matrix_1 = fc_topo(10) 
    W_matrix_2 = ring_topo(10)

    model_kwargs = {"input_dim" : 7, "output_dim": 1}
    optimiser_kwargs = {"lr" : 0.001} #specify keyword args for model

    graph_kwargs = {"model_kwargs": model_kwargs, #pass model kwargs
        "optimiser_kwargs" : optimiser_kwargs,
        "criteria" : nn.MSELoss, #specify loss function for each node
        "model" : model_lr, #specify model class handle
        "optimiser" : torch.optim.SGD, #specify global optimiser
        "batch_size" : 10 #specify batch size for each node
        }

    draw_graph(fc_topo(10))

    # graph_1 = graph(data, W_matrix_1, iid=True, **graph_kwargs)
    # graph_2 = graph(data, W_matrix_2, iid=True, **graph_kwargs)
    # graph_3 = graph(data, W_matrix_2, iid=True, **graph_kwargs)

    # graph_1.run(mixing_steps=1, local_steps=1, iters=20) 
    # graph_2.run(mixing_steps=1, local_steps=1, iters=20)
    # graph_3.run(mixing_steps=4, local_steps=1, iters=20)

    # t = [i for i in range(20)]
    # plt.plot(t, graph_1.losses, label="5")
    # plt.plot(t, graph_2.losses, label="10")
    # plt.plot(t, graph_3.losses, label="10-more-mixing")
    
    # plt.legend()
    # plt.show() 