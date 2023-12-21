import numpy as np
import pandas as pd
import torch
import torch_geometric


def read_data(file_path):
    # determine the number of columns
    df = pd.read_csv(file_path)
    num_cols = len(df.columns) - 1

    # read data
    data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=np.arange(1, num_cols + 1))
    
    return data


def split(expression: torch.Tensor, target: torch.Tensor, graph: torch_geometric.data.Data, encode_dim: int = 0):
    """
    Split the data into training, validation and test sets

    :param expression: expression data, with shape (num_sample, num_feature)
    :param target: diagnosis data, one-hot encoded, with shape (num_sample, num_class)
    :param graph: graph structure
    :param encode_dim: dimension of positional encoding
    :return: training, validation and test sets
    """
    train_list, test_list, valid_list = [], [], []
    num_sample = len(target)
    train_index, val_index = int(num_sample*0.8), int(num_sample*0.9)

    for i in range(num_sample):
        x = expression[i]
        x = torch.unsqueeze(x, dim=1).float()
        if encode_dim > 0:
            dimension = len(x)
            positional_encoder = torch.rand(dimension, encode_dim).float()
            x = torch.cat((x, positional_encoder), 1)
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

