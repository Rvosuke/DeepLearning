import torch
import torch_geometric
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # Initialisation of self.convs, 
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        # Construct all convs
        self.convs = torch.nn.ModuleList()

        # construct all bns
        self.bns = torch.nn.ModuleList()

        #For the first layer, we go from dimensions input -> hidden
        #For middle layers we go from dimensions hidden-> hidden
        #For the end layer we go from hidden-> output

        for l in range(num_layers):
          if l==0: #change input output dims accordingly
            self.convs.append(GCNConv(input_dim, hidden_dim))
          elif l == num_layers-1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
          else:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
          if l < num_layers-1: 
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.last_conv = GCNConv(hidden_dim, output_dim)
        self.log_soft = torch.nn.LogSoftmax()

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_weight):
        # This function that takes the feature tensor x and
        # edge_index tensor adj_t, and edge_weight and returns the output tensor.

        out = None

        for l in range(len(self.convs)-1):
          x = self.convs[l](x, adj_t, edge_weight)
          x = self.bns[l](x)
          x = F.relu(x)
          x = F.dropout(x, training=self.training)

        x = self.last_conv(x, adj_t, edge_weight)
        if self.return_embeds is True:
          out = x
        else: 
          out = self.log_soft(x)

        return out


class GCN_Graph(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Node embedding model, initially input_dim=input_dim, output_dim = hidden_dim
        self.gnn_node = GCN(input_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)
        # Note that the input_dim and output_dim are set to hidden_dim
        # for subsequent layers
        self.gnn_node_2 = GCN(hidden_dim, hidden_dim,
        hidden_dim, num_layers, dropout, return_embeds=True)

        # Set up pooling layer using ASAPool
        self.asap = torch_geometric.nn.pool.ASAPooling(in_channels = 256, ratio = 0.5, dropout = 0.1, negative_slope = 0.2, add_self_loops = False)

        # Initialize self.pool as a global mean pooling layer
        self.pool = torch_geometic.nn.global_mean_pool

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # This function takes as input a 
        # mini-batch of graphs (torch_geometric.data.Batch) and 
        # returns the predicted graph property for each graph. 
        #
        # Since we are predicting graph level properties,
        # the output will be a tensor with dimension equaling
        # the number of graphs in the mini-batch

    
        # Extract important attributes of our mini-batch
        x, edge_index, batch, edge_weight = batched_data.x, batched_data.edge_index, batched_data.batch, batched_data.edge_attr
        embed = x
        out = None

        ## Note:
        ## 1. We construct node embeddings using existing GCN model
        ## 2. We use the ASAPool module for soft clustering into a coarser graph representation. 
        ## For more information please refere to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.pool.ASAPooling
        ## 3. After two cycles of this, we use the global pooling layer to aggregate features for each individual graph
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        ## 4. We use a linear layer to predict each graph's property
        num_graphs = int(len(batch)/31)
        post_GCN_1 = self.gnn_node(embed, edge_index, edge_weight)
        post_pool_1 = self.asap(post_GCN_1, edge_index)
        post_GCN_2 = self.gnn_node_2(post_pool_1[0], post_pool_1[1], post_pool_1[2])
        post_pool_2 = self.asap(post_GCN_2, post_pool_1[1])
        ultimate_gcn = self.gnn_node_2(post_pool_2[0], post_pool_2[1], post_pool_2[2])

        glob_pool = self.pool(ultimate_gcn, post_pool_2[3], num_graphs)  
        out = self.linear(glob_pool)    

        return out
