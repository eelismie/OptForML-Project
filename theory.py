import torch
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt

from modules.topos import *
from modules.graph import *


def toy_data(n_workers):
    x = torch.randn(3,1)

    A = torch.randn(n_workers, 3, 3)

    b = torch.bmm(A, torch.cat([x for _ in range(n_workers)], dim=1).T.unsqueeze(-1))

    return A, x, b


if __name__ == "__main__":
    np.random.seed(4)
    torch.manual_seed(4)

    n_workers = 100
    A, x, b = toy_data(n_workers)

    W = random_topo(n_workers)
    W_mh = MH_weights(W)

    print("edge proportion vs fc net: ", W.sum() / n_workers ** 2)

    model_kwargs = {"input_dim" : 3, "output_dim": 1}
    optimiser_kwargs = {"lr" : 0.35}

    # cutoff learning rate is 0.5
    graph_kwargs = {"model_kwargs": model_kwargs, #pass model kwargs
        "optimiser_kwargs" : optimiser_kwargs,
        "criteria" : nn.MSELoss, #specify loss function for each node
        "model" : model_lr, #specify model class handle
        "optimiser" : torch.optim.SGD, #specify global optimiser
        "batch_size" : 3 #specify batch size for each node
    }

    graph = graph((A, b), W_mh, iid=True, toy_example=True, **graph_kwargs)


    Lh = max([x.lipschitz for x in graph.nodes])
    lambda_n = np.linalg.eig(W_mh)[0].min()
    lr = min((1 + lambda_n) / Lh, 1 / Lh)

    print("lr: ", lr, "lambda_n: ", lambda_n, "Lh: ", Lh)

    graph.run(mixing_steps = 1,
              local_steps = 1,
              iters= 100)

    # plt.plot(range(100), graph.losses)
    # plt.show()
