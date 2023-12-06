import torch
import networkx as nx
import numpy as np
import torch_geometric.utils
from torch_geometric.data import Data, DataLoader


# Save the values into an adjacency matrix
adj = np.loadtxt(open("adjacency_matrix.csv", "rb"), delimiter=",", skiprows=1, usecols=np.arange(2, 53))

# Set up graph from adjacency matrix and assign protein name labels
G = nx.from_numpy_matrix(adj, parallel_edges=False, create_using=None)

num_sample = len(df)

# Save the protein expression levels into a matrix
expression_mat = numpy.loadtxt(open("log_transformed_ADNI_expression_data_with_covariates.csv", "rb"), delimiter=",", skiprows=1, usecols=numpy.arange(16, 67))
# print(expression_mat[50,:])

x_tensor = torch.from_numpy(np.array(df)).float()
target_tensor = torch.Tensor(data.target).long()
adj_tensor = torch.from_numpy(np.array(adj_matrix))
sm_convert = torch_geometric.utils.from_networkx(sm)

positional_encoder = torch.rand(30, 6).float()

split_idx = {
    'train': torch.tensor(np.arange(0, 400)),
    'valid': torch.tensor(np.arange(400, 500)),
    'test': torch.tensor(np.arange(500, 569)),
}

train_list = []
test_list = []
valid_list = []
for i in range(len(target_tensor)):
  x_yeet = x_tensor[i,:]
  x_scalar = torch.t(torch.reshape(x_yeet, (1, len(x_yeet)))).float()
  x = torch.cat((x_scalar, positional_encoder), 1)
  y = target_tensor[i]
  if i in split_idx['train']:
    train_list.append(Data(x=x, y=y, edge_index=sm_convert.edge_index, edge_attr=sm_convert.weight))
  elif i in split_idx['valid']:
    valid_list.append(Data(x=x, y=y, edge_index=sm_convert.edge_index, edge_attr=sm_convert.weight))
  elif i in split_idx['test']:
    test_list.append(Data(x=x, y=y, edge_index=sm_convert.edge_index, edge_attr=sm_convert.weight))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader = DataLoader(train_list, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_list, batch_size=8, shuffle=False)
test_loader = DataLoader(test_list, batch_size=8, shuffle=False)
#%%
args = {
    'device': device,
    'num_layers': 5,
    'hidden_dim': 256,
    'dropout': 0.3,
    'lr': 1e-3,
    'epochs': 50,
}
