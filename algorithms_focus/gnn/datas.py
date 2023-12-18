import torch
import torch_geometric
import numpy as np
import pandas as pd
import networkx as nx


def read_data(file_path, is_adjacency=False):
    # determine the number of columns
    df = pd.read_csv(file_path)
    num_cols = len(df.columns) - 1

    # read data
    data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=np.arange(1, num_cols + 1))
    
    return data


def split(expression, target, graph, encode_dim=0):
    train_list, test_list, valid_list = [], [], []
    num_sample = len(target)
    train_index, val_index = int(num_sample*0.8), int(num_sample*0.9)

    for i in range(num_sample):
        x = expression[i]
        if encode_dim > 0:
            positional_encoder = torch.rand(31, encode_dim).float()
            x_scalar = torch.unsqueeze(x, dim=1).float()
            x = torch.cat((x_scalar, positional_encoder), 1)
        y = target[i]

        params = {"x": x, "y": y, "edge_index": graph.edge_index, "edge_attr": graph.weight}
        data = torch_geometric.data.Data(**params)
        if i in torch.arange(0, train_index):
            train_list.append(data)
        elif i in torch.arange(train_index, val_index):
            valid_list.append(data)
        elif i in torch.arange(val_index, num_sample):
            test_list.append(data)

    return train_list, valid_list, test_list

