"Utility functions for preprocessing data."
import re
import pandas as pd
from IPython import embed
import torch
import numpy as np
import networkx as nx

def preprocess_car_data(df):
    """Apply simple preprocessing and return cleaned data"""
    df.dropna(inplace=True)

    df["name"] = df["name"].str.split(" ").str[0]
    df["engine"] = df["engine"].str.split(" ").str[0]
    df["max_power"] = df["max_power"].str.split(" ").str[0]
    df["mileage"] = df["mileage"].str.split(" ").str[0]

    di = {"First Owner": 0.0 , "Second Owner": 1.0,"Third Owner": 2.0,
          "Fourth & Above Owner": 3.0, "Test Drive Car": 4.0}
    df.replace({"owner": di}, inplace=True)

    df.drop(["torque", "transmission", "seller_type", "fuel"], axis=1, inplace=True)

    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    df["selling_price"] = df["selling_price"].divide(100)

    return df


def car_train_test(df, train_split=0.8):
    """Create train and test datasets"""
    df = df.sample(frac=1).reset_index(drop=True)
    df.describe()

    split_ind = int(train_split * len(df))
    df_train = df.iloc[:split_ind]
    df_test = df.iloc[split_ind:]

    df_train.iloc[:, 1:] = df_train.iloc[:, 1:].astype(float)
    df_test.iloc[:, 1:] = df_test.iloc[:, 1:].astype(float)

    return df_train, df_test

def car_to_torch(df_train, df_test):

    """ preprocess car dataset - from df to torch """

    data = (df_train, df_test)

    test_data = data[1].drop("name", axis=1)
    x_test = torch.from_numpy(np.array(test_data.drop("selling_price", axis=1))).float()
    y_test = torch.from_numpy(np.array(test_data["selling_price"])).float()

    x_test, y_test

    train_data = data[0]
    # sort by car brand so each node is assigned mostly one brand (or whatever else if we use another dataset)
    train_data.sort_values("name", inplace=True)
    x_train = torch.from_numpy(np.array(train_data.drop(["selling_price", "name"], axis=1))).float()
    y_train = torch.from_numpy(np.array(train_data["selling_price"])).float()

    mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
    x_train.sub_(mu).div_(sd)

    return x_train, y_train, x_test, y_test


def synthetic_data(samples, in_features):
    """ sample points from a plane """

    w = torch.empty((in_features, 1)).uniform_()
    X = torch.empty((samples, in_features)).uniform_()

    Y = torch.matmul(X, w) + 0.1*torch.empty((samples, 1)).normal_()
    return X, Y, w

def draw_graph(w_matrix):
    """ use networkx to visualize the graph topology """

    G = nx.Graph()
    copy = np.copy(w_matrix)
    for i in range(w_matrix.shape[0]):
        copy[i,i] = 0
        row = copy[i,:]
        nonzero = np.nonzero(row) 
        for j in nonzero[0]:
            G.add_edge(i + 1, j + 1)
        
    nx.draw(G)
    plt.show()

def glob_optimum_lr(X, y):
    w = np.linalg.inv(X.T@X)@X.T@y
    diff = X@w - y
    optimum = (diff*diff).mean()
    return optimum




