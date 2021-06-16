#!/bin/python3
"""Main script which trains a simulated distributed model given some argument options."""
import torch
import pandas as pd
from torch import nn
from matplotlib import pyplot as plt
import numpy as np


from modules.data import get_data
from modules.options import parse_args
from modules.utils import open_csv, get_bandwidth
from modules.graph import graph, model_lr
from modules.topos import ring_topo, fc_topo, random_topo, MH_weights


if __name__ == '__main__':
    # retrieve command line args
    opt = parse_args()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.iid.lower() == "false":
        opt.iid = False
    else:
        opt.iid = True

    # optional csv output
    if opt.csv:
        opt.csv = open_csv(opt.csv,
                header='topo,nodes,lr,batch_size,mixing_steps,local_steps,loss,comms')

    if opt.topo == "fc":
        W_matrix = fc_topo(opt.nodes)
    elif opt.topo == "random":
        W_matrix = MH_weights(random_topo(opt.nodes))
    elif opt.topo == "ring":
        W_matrix = ring_topo(opt.nodes)
    else:
        raise ValueError(f'Topology "{opt.topo}" is not valid.')

    x_train, y_train = get_data(opt.num_samples)

    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
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

    graph_1 = graph(data, W_matrix, iid=opt.iid, **graph_kwargs)
    graph_1.run(mixing_steps=opt.mixing_steps,
                local_steps=opt.local_steps,
                iters=opt.iters)

    # optional csv output
    if opt.csv:
        topo_BW = get_bandwidth(W_matrix,
                                graph_1.nodes[0].model.parameters(),
                                opt.mixing_steps)
        c = 0
        for l in graph_1.losses:
            c += topo_BW
            opt.csv.write(f'{opt.topo},{opt.nodes},{opt.lr},{opt.batch_size},'
                          f'{opt.mixing_steps},{opt.local_steps},{l},{c}\n')
        opt.csv.close()
