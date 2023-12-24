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
        self.gcn_layers = torch.nn.ModuleList()
        self.dropout = dropout
        # self.gnn_node1 = GCNBase(input_dim, hidden_dim, hidden_dim, gcn_layers, dropout, return_embeds=True)
        # self.gnn_node2 = GCNBase(hidden_dim, hidden_dim, hidden_dim, gcn_layers, dropout, return_embeds=True)
        for i in range(gcn_base_layers):
            if i == 0:
                self.gcn_layers.append(torch_geometric.nn.GraphConv(input_dim, hidden_dim))
            else:
                self.gcn_layers.append(torch_geometric.nn.GraphConv(hidden_dim, hidden_dim))
        self.asap = torch_geometric.nn.pool.ASAPooling(256, dropout=dropout)
        self.linear1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)
        self.log_soft = torch.nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        for gcn in self.gcn_layers:
            gcn.reset_parameters()
        self.asap.reset_parameters()
        # self.linear1.reset_parameters()
        # self.linear2.reset_parameters()

    def forward(self, data):
        readouts = []
        post = (data.x, data.edge_index, data.edge_attr, data.batch)

        for i in range(len(self.gcn_layers)):
            post_gcn = self.gcn_layers[i](post[0], post[1], post[2])
            post_gcn = torch.relu(post_gcn)
            post = self.asap(post_gcn, post[1], post[2], post[3])
            post_readout = readout(post[0], post[3])
            readouts.append(post_readout)
        out = sum(readouts)

        out = self.linear1(out)
        out = torch.relu(out)
        out = torch.dropout(out, self.dropout, train=self.training)
        out = self.linear2(out)
        # out = self.log_soft(out)
        return out

    def __repr__(self):
        return self.__class__.__name__


def readout(x, batch):
    # num_graphs = int(len(batch)/31)
    # x_mean = torch_scatter.scatter_mean(x, batch, dim=0)
    # x_max, _ = torch_scatter.scatter_max(x, batch, dim=0)
    x_mean = torch_geometric.nn.global_mean_pool(x, batch)
    x_max = torch_geometric.nn.global_max_pool(x, batch)
    out = torch.cat((x_mean, x_max), dim=-1)
    return out


class ASAP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.3):
        super().__init__()
        self.conv1 = torch_geometric.nn.GraphConv(input_dim, hidden_dim, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.extend([
            torch_geometric.nn.GraphConv(hidden_dim, hidden_dim, aggr='mean')
            for _ in range(num_layers - 1)
        ])
        self.pools.extend([
            torch_geometric.nn.ASAPooling(hidden_dim, dropout=dropout)
            for _ in range(num_layers // 2)
        ])
        self.bns.extend([
            torch.nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers - 1)
        ])
        self.jump = torch_geometric.nn.JumpingKnowledge(mode='cat')
        self.lin1 = torch.nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None
        x = torch.relu(self.conv1(x, edge_index))
        xs = [torch_geometric.nn.global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = torch.relu(x)
            xs += [torch_geometric.nn.global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = torch.relu(self.lin1(x))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.lin2(x)
        # x = torch.log_softmax(x, dim=-1)
        return x

    def __repr__(self):
        return self.__class__.__name__
