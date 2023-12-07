import torch
import torch_geometric


class GCNBase(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        # Initialisation of self.convs, self.bns, and self.softmax.
        super(GCNBase, self).__init__()
        self.softmax = None  # The log softmax layer
        self.convs = torch.nn.ModuleList()  # Construct all convs
        self.bns = torch.nn.ModuleList()  # construct all bns A list of 1D batch normalization layers

        for layer in range(num_layers - 1):
            if layer == 0:  # For the first layer, we go from dimensions input -> hidden
                self.convs.append(torch_geometric.nn.GCNConv(input_dim, hidden_dim))
            else:  # For middle layers we go from dimensions hidden-> hidden
                self.convs.append(torch_geometric.nn.GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        # For the end layer we go from hidden-> output
        self.last_conv = torch_geometric.nn.GCNConv(hidden_dim, output_dim)
        self.log_soft = torch.nn.LogSoftmax()
        self.dropout = dropout
        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight):
        for i in range(len(self.convs)):
            x = self.convs[i](x, adj_t, edge_weight)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = torch.dropout(x, self.dropout, train=self.training)
        x = self.last_conv(x, adj_t, edge_weight)

        if self.return_embeds:
            return x
        else:
            return self.log_soft(x)


class GCNGraph(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCNGraph, self).__init__()
        self.gnn_node1 = GCNBase(input_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.gnn_node2 = GCNBase(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.asap = torch_geometric.nn.pool.ASAPooling(256, 0.5, dropout=0.1, negative_slope=0.2, add_self_loops=False)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.gnn_node1.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x):
        num_graphs = int(len(x.batch) / 31)
        x = self.gnn_node1(x.x, x.edge_index, x.edge_attr)
        x = self.asap(x)
        x = self.gnn_node2(x.x, x.edge_index, x.edge_attr)
        x = self.asap(x)
        x = self.gnn_node2(x.x, x.edge_index, x.edge_attr)

        x = torch_geometric.nn.global_mean_pool(x, x.batch, num_graphs)
        x = self.linear(x)

        return x
