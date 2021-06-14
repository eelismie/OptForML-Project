#!/bin/python3
"""Main script which trains a simulated distributed model given some argument options."""
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from modules.data import get_data
from modules.options import parse_args
from modules.graph import graph, model_lr
from modules.topos import ring_topo, fc_topo, random_topo, MH_weights


if __name__ == '__main__':
    # retrieve command line args
    opt = parse_args()

    if opt.topo == "fc":
        W_matrix = fc_topo(opt.nodes)
    elif opt.topo == "random":
        W_matrix = random_topo(opt.nodes)
    elif opt.topo == "mh":
        # W_matrix = MH_weights(opt.nodes)
        pass
    elif opt.topo == "ring":
        W_matrix = ring_topo(opt.nodes)
    else:
        raise ValueError(f'Topology "{opt.topo}" is not valid.')

    X, y = get_data(opt.num_samples)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    # exit(0)

    data = (x_train, y_train)

    model_kwargs = {"input_dim" : 2, "output_dim": 1}
    optimiser_kwargs = {"lr" : opt.lr} #specify keyword args for model

    graph_kwargs = {"model_kwargs": model_kwargs, #pass model kwargs
        "optimiser_kwargs" : optimiser_kwargs,
        "criteria" : nn.MSELoss, #specify loss function for each node
        "model" : model_lr, #specify model class handle
        "optimiser" : torch.optim.SGD, #specify global optimiser
        "batch_size" : opt.batch_size #specify batch size for each node
    }


    graph_1 = graph(data, W_matrix, iid=True, **graph_kwargs)
    graph_1.run(mixing_steps=opt.mixing_steps,
                local_steps=opt.local_steps,
                iters=opt.epochs)
