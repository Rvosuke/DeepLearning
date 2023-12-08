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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCNGraph, self).__init__()
        self.gnn_node1 = GCNBase(input_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.gnn_node2 = GCNBase(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout, return_embeds=True)
        self.asap = torch_geometric.nn.pool.ASAPooling(256, 0.5, dropout=0.1, negative_slope=0.2, add_self_loops=False)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.gnn_node1.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):
        num_graphs = int(len(data.batch) / 31)

        post_gcn1 = self.gnn_node1(data.x, data.edge_index, data.edge_attr)
        post_pool1 = self.asap(post_gcn1, data.edge_index)
        readout1 = torch_geometric.nn.global_mean_pool(post_pool1[0], post_pool1[3], num_graphs)

        post_gcn2 = self.gnn_node2(post_pool1[0], post_pool1[1], post_pool1[2])
        post_pool2 = self.asap(post_gcn2, post_pool1[1])
        readout2 = torch_geometric.nn.global_mean_pool(post_pool2[0], post_pool2[3], num_graphs)

        post_gcn3 = self.gnn_node2(post_pool2[0], post_pool2[1], post_pool2[2])
        post_pool3 = self.asap(post_gcn3, post_pool2[1])
        readout3 = torch_geometric.nn.global_mean_pool(post_pool3[0], post_pool3[3], num_graphs)


        # +5layers
        out = readout1 + readout2 + readout3
        out = self.linear(out)

        return out
