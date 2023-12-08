import torch
import torch_geometric
import numpy as np
import networkx as nx

from torch_geometric.data import Data


adj = np.loadtxt(open("adjacency_matrix.csv", "rb"), delimiter=",", skiprows=1, usecols=np.arange(1, 32))
G = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)
expression_matrix = np.loadtxt(open("expression.csv", "rb"), delimiter=",", skiprows=1, usecols=np.arange(1, 32))
binary_diagnosis = np.loadtxt(open("target.csv", "rb"), delimiter=",", skiprows=1, usecols=np.arange(1, 2))

# prevent target feature from leaking information and construct virtual nodes
in_edge = adj[30].astype(bool)
for sample in expression_matrix:
    sample[30] = np.mean(sample[in_edge])

x_tensor = torch.from_numpy(expression_matrix).float()
diagnosis_tensor = torch.Tensor(binary_diagnosis).long()
# adj_tensor = torch.from_numpy(adj)
G_convert = torch_geometric.utils.from_networkx(G)


def split(expression, target, graph, encode_dim=3):
    if isinstance(expression, np.ndarray):
        expression = torch.from_numpy(expression).float()
        target = torch.Tensor(target).long()
    elif isinstance(expression, torch.Tensor):
        pass
    else:
        raise TypeError(f'expression and target should be ndarray or tensor, {expression.__class__}{target.__class__}')
    train_list = []
    test_list = []
    valid_list = []
    # num_feature = len(expression[0])
    num_sample = len(target)
    train_index = int(num_sample*0.8)
    val_index = int(num_sample*0.9)
    positional_encoder = torch.rand(31, encode_dim).float()
    for i in range(num_sample):
        x_yeet = expression[i]
        x_scalar = torch.unsqueeze(x_yeet, dim=1).float()
        x = torch.cat((x_scalar, positional_encoder), 1)
        y = target[i]
        if i in torch.arange(0, train_index):
            train_list.append(Data(x=x, y=y, edge_index=graph.edge_index, edge_attr=graph.weight))
        elif i in torch.arange(train_index, val_index):
            valid_list.append(Data(x=x, y=y, edge_index=graph.edge_index, edge_attr=graph.weight))
        elif i in torch.arange(val_index, num_sample):
            test_list.append(Data(x=x, y=y, edge_index=graph.edge_index, edge_attr=graph.weight))

    return train_list, valid_list, test_list


data_split = split(x_tensor, diagnosis_tensor, G_convert)
