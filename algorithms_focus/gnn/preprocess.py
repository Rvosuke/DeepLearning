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
    Split the data into training, validation and test sets after shuffling.

    :param expression: expression data, with shape (num_sample, num_feature)
    :param target: diagnosis data, one-hot encoded, with shape (num_sample, num_class)
    :param graph: graph structure
    :param encode_dim: dimension of positional encoding
    :return: training, validation and test sets
    """
    train_list, test_list, valid_list = [], [], []
    num_sample = len(target)
    
    # Generate shuffled indices
    # shuffled_indices = torch.randperm(num_sample)
    
    # Determine split sizes
    train_index, val_index = int(4318*0.9), int(4318)

    for idx in range(num_sample):
        x = expression[idx]
        x = torch.unsqueeze(x, dim=1).float()
        if encode_dim > 0:
            dimension = len(x)
            positional_encoder = torch.rand(dimension, encode_dim).float()
            x = torch.cat((x, positional_encoder), 1)
        y = target[idx]

        params = {"x": x, "y": y, "edge_index": graph.edge_index, "edge_attr": graph.weight}
        data = torch_geometric.data.Data(**params)
        
        if idx < train_index:
            train_list.append(data)
        elif train_index <= idx < val_index:
            valid_list.append(data)
        else:
            test_list.append(data)

    return train_list, valid_list, test_list

