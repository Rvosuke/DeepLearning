import networkx as nx
import numpy as np
import warnings
import argparse
import torch
import torch_geometric

from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datas import split, read_data
from models import GCNGraph, ASAP
from utils import train, evaluate

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=256)
    parser.add_argument("-l", "--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=300)
    # parser.add_argument("-p", "--pos", help="whether to use positional encoding", action="store_true")
    parser.add_argument("-d", "--dropout", help="dropout rate", type=float, default=0.5)
    parser.add_argument("-g", "--gcn_layers", help="number of GCN layers", type=int, default=5)
    parser.add_argument("-bl", "--gcn_base_layers", help="number of GCN base layers", type=int, default=3)
    parser.add_argument("-i", "--in_channels", help="number of input channels", type=int, default=1)
    parser.add_argument("-o", "--out_channels", help="number of output channels", type=int, default=2)
    parser.add_argument("-n", "--hidden_dim", help="number of hidden dimensions", type=int, default=256)
    args = parser.parse_args()
    return args


def main():
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adj, expression, diagnosis = read_data("adjacency_matrix.csv"), read_data(
        "expression.csv"), read_data("target.csv")
    # prevent target feature from leaking information and construct virtual nodes
    in_edge = adj[:, -1].astype(bool)  # the last Colum of the adjacency matrix is the virtual node
    for sample in expression:
        sample[-1] = np.mean(sample[in_edge])
    G = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)
    x_tensor = torch.tensor(expression, dtype=torch.float, device=device)
    diagnosis_tensor = torch.tensor(diagnosis, dtype=torch.long, device=device)
    diagnosis_tensor = torch.nn.functional.one_hot(diagnosis_tensor, num_classes=2)
    G_convert = torch_geometric.utils.from_networkx(G).cuda()

    a = set_args()
    # a = vars(s)

    args = {
        'batch_size': a.batch_size,
        # 'no_pos': not a.pos,  # if False, positional encoding will not be used
        'lr': a.lr,
        'epochs': a.epochs,
    }
    model_params = {
        'input_dim': a.in_channels,
        # 'in_channels': 1 if args['no_pos'] else 4,
        'hidden_dim': a.hidden_dim,
        'output_dim': a.out_channels,
        'num_layers': a.gcn_base_layers,
        'dropout': a.dropout,
    }
    # encode_dim = 0 if args['no_pos'] else gcn_params['in_channels']
    encode_dim = 0

    trains, vals, tests = split(x_tensor, diagnosis_tensor, G_convert, encode_dim)
    train_loader = DataLoader(trains, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    valid_loader, test_loader = DataLoader(vals), DataLoader(tests)
    # model = GCNGraph(**gcn_params)
    model = ASAP(**model_params)
    model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
    # threshold=0.01, threshold_mode='rel', eps=1e-8)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.CrossEntropyLoss()

    # Keep the best model
    b_acc = 0
    b_model = None
    print("Hello World!")
    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, device, train_loader, optimizer, loss_fn)
        scheduler.step()
        *rating, b_acc, b_model = evaluate(model, device, valid_loader, b_acc, b_model)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/train', rating[0], epoch)
        writer.add_scalar('Precision/train', rating[1], epoch)
        writer.add_scalar('Sensitivity/train', rating[2], epoch)
        writer.add_scalar('Specificity/train', rating[3], epoch)
        writer.add_scalar('AUC/test', rating[6], epoch)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, '
              f'Sensitivity: {rating[2]:.4f}, Specificity: {rating[3]:.4f}, AUC: {rating[6]:.4f}')

    *rating, b_acc, b_model = evaluate(b_model, device, test_loader, b_acc, b_model)
    print(f'Test Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, Sensitivity: {rating[2]:.4f}, '
          f'Specificity: {rating[3]:.4f}, fpr: {rating[4]}, tpr: {rating[5]}, AUC: {rating[6]:.4f}')


if __name__ == '__main__':
    main()
