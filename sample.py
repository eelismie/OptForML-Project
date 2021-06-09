import torch
import torch.nn as nn
import torchvision
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torchvision.datasets as dsets

class node():
    def __init__(self, model, data_x, data_y, criteria):
        self.model = model
        self.X = data_x
        self.Y - data_y
        self.criteria = criteria

    def parameters(self):
        return self.model.parameters()

    def forward_backward(self):
        out = self.model(self.X)
        l = self.criteria(out, self.Y)
        l.backward()

class graph():
    def __init__(self, W_matrix, data, model, loss, **kwargs):
        self.W = W_matrix
        self.nodes = [node(model(**kwargs), data[0], data[1], loss) for i in range(W_matrix.shape[0])] #initialize models

        #TODO: partition data among nodes instead of passing data[0] and data[1]
        #TODO: optimization step and mixing step
        #TODO: check if this initialization actually works

    def run(self, mixing_steps=1, local_steps=1, iters=100):
        #run optimization for 1000 total iterations with a defined number of mixing steps and local update steps
        pass


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






samples= 1000
tot_samples = samples*2
features = 784
classes = 2
lips = 0.25/float(samples)
lr = 1.0/lips

dataset = torchvision.datasets.MNIST(root = "data",train=True,download=True )

idx_0 = (dataset.train_labels==0)
idx_1 = (dataset.train_labels==1)

x = torch.cat([dataset.train_data[idx_0][:samples], dataset.train_data[idx_1][:samples]] ).reshape(tot_samples,-1)
y = torch.cat([dataset.train_labels[idx_0][:samples], dataset.train_labels[idx_1][:samples]])

#compute spectral norm
eig = np.linalg.eigvals(x.numpy().T @ x.numpy()).max()
lips = eig*lips
print('eig :', eig)
lr = 1. / lips
print('lips :', lips, 'lr :', lr)

x = x.type(torch.FloatTensor)
y = y.type(torch.LongTensor)
model = nn.Linear(features, 2)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)

losses = []
iters = []
for i in range(500):
    out = model(x)
    l = criterion(out, y)
    l.backward()
    optim.step()
    optim.zero_grad()
    losses.append(l.item())
    iters.append(i)
    if i % 100 == 0:
        print('i :', i, 'loss :', l.item())

# we should see a linear decay in the log-log plot
plt.yscale('log')
plt.xscale('log')
plt.scatter(iters, losses)
plt.show()


