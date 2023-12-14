import warnings
import argparse
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from datas import split, read_data
from models import GCNGraph
from utils import train, evaluate


warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("-l", "--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=100)
    # parser.add_argument("-p", "--pos", help="whether to use positional encoding", action="store_true")
    parser.add_argument("-d", "--dropout", help="dropout rate", type=float, default=0.3)
    parser.add_argument("-g", "--gcn_layers", help="number of GCN layers", type=int, default=5)
    parser.add_argument("-b", "--gcn_base_layers", help="number of GCN base layers", type=int, default=3)
    parser.add_argument("-i", "--in_channels", help="number of input channels", type=int, default=1)
    parser.add_argument("-o", "--out_channels", help="number of output channels", type=int, default=1)
    parser.add_argument("-n", "--hidden_dim", help="number of hidden dimensions", type=int, default=256)
    args = parser.parse_args()
    return args

def main():
    adj, expression, diagnosis = read_data("adjacency_matrix.csv", is_adjacency=True), read_data("expression.csv"), read_data("target.csv")
    
    # prevent target feature from leaking information and construct virtual nodes
    in_edge = adj[-1].astype(bool)  # the last row of the adjacency matrix is the virtual node
    for sample in expression:
        sample[-1] = np.mean(sample[in_edge])
    G = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)

    x_tensor = torch.from_numpy(expression).float()
    diagnosis_tensor = torch.Tensor(diagnosis).long()
    G_convert = from_networkx(G)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_args = set_args()
    a = vars(set_args)

    args = {
        'batch_size': a.batch_size,
        # 'no_pos': not a.pos,  # if False, positional encoding will not be used
        'lr': a.lr,
        'epochs': a.epochs,
    }
    gcn_params = {
        'in_channels': a.in_channels,
        # 'in_channels': 1 if args['no_pos'] else 4,
        'hidden_dim': a.hidden_dim,
        'out_channels': a.out_channels,
        'gcn_layers': a.gcn_layers,
        'gcn_base_layers': a.gcn_base_layers,
        'dropout': a.dropout,
    }
    encode_dim = 0 if args['no_pos'] else gcn_params['in_channels']

    train, val, test = split(x_tensor, diagnosis_tensor, G_convert, encode_dim)
    train_loader = DataLoader(train, batch_size=args['batch_size'], shuffle=True)
    valid_loader, test_loader = DataLoader(val), DataLoader(test)
    model = GCNGraph(**gcn_params).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Keep the best model
    b_acc = 0
    b_model = None
    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, device, train_loader, optimizer, loss_fn)
        scheduler.step()
        *rating, b_acc, b_model = evaluate(model, device, valid_loader, b_acc, b_model)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, '
              f'Sensitivity: {rating[2]:.4f}, Specificity: {rating[3]:.4f}, AUC: {rating[6]:.4f}')

    *rating, b_acc, b_model = evaluate(b_model, device, test_loader, b_acc, b_model)
    print(f'Test Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, Sensitivity: {rating[2]:.4f}, '
          f'Specificity: {rating[3]:.4f}, fpr: {rating[4]}, tpr: {rating[5]}, AUC: {rating[6]:.4f}')


if __name__ == '__main__':
    main()
