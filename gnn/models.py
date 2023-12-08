import torch
import torch_geometric


class GCNBase(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        super(GCNBase, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

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
    def __init__(self, input_dim, hidden_dim, output_dim, gcn_layers=5, gcn_base_layers=3, dropout=0.3):
        super(GCNGraph, self).__init__()
        self.gcn_base_layers = gcn_base_layers
        self.gnn_node1 = GCNBase(input_dim, hidden_dim, hidden_dim, gcn_layers, dropout, return_embeds=True)
        self.gnn_node2 = GCNBase(hidden_dim, hidden_dim, hidden_dim, gcn_layers, dropout, return_embeds=True)
        self.asap = torch_geometric.nn.pool.ASAPooling(256, 0.5, dropout=dropout, negative_slope=0.2, add_self_loops=False)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.gnn_node1.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):
        # num_nodes_per_graph = data.batch.bincount()
        readouts = []
        post_pool = (data.x, data.edge_index, data.edge_attr, None)

        for i in range(self.gcn_base_layers):
            if i == 0:
                post_gcn = self.gnn_node1(post_pool[0], post_pool[1], post_pool[2])
            else:
                post_gcn = self.gnn_node2(post_pool[0], post_pool[1], post_pool[2])
            post_pool = self.asap(post_gcn, post_pool[1])
            readout = torch_geometric.nn.global_mean_pool(post_pool[0], post_pool[3], data.num_graphs)
            readouts.append(readout)

        return self.linear(sum(readouts))
