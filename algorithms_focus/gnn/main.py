import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import torch
import torch_geometric

from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import split, read_data
from models import GCNGraph, ASAP
from operator import train, evaluate

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("-l", "--lr", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=256)
    # parser.add_argument("-p", "--pos", help="whether to use positional encoding", action="store_true")
    parser.add_argument("-d", "--dropout", help="dropout rate", type=float, default=0.5)
    parser.add_argument("-g", "--gcn_layers", help="number of GCN layers", type=int, default=5)
    parser.add_argument("-bl", "--gcn_base_layers", help="number of GCN base layers", type=int, default=4)
    parser.add_argument("-i", "--in_channels", help="number of input channels", type=int, default=1)
    parser.add_argument("-o", "--out_channels", help="number of output channels", type=int, default=2)
    parser.add_argument("-n", "--hidden_dim", help="number of hidden dimensions", type=int, default=256)
    args = parser.parse_args()
    return args


def pre_data(device=torch.device('cuda')):
    adj, expression, diagnosis = read_data("adjacency_matrix.csv"), read_data(
        "expression.csv"), read_data("target.csv")
    # prevent target feature from leaking information and construct virtual nodes
    in_edge = adj[-1].astype(bool)  # the last Colum of the adjacency matrix is the virtual node
    for sample in expression:
        sample[-1] = np.mean(sample[in_edge])
    G = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)
    x_tensor = torch.tensor(expression, dtype=torch.float, device=device)
    diagnosis_tensor = torch.tensor(diagnosis, dtype=torch.long, device=device)
    diagnosis_tensor = torch.nn.functional.one_hot(diagnosis_tensor, num_classes=2)
    G_convert = torch_geometric.utils.from_networkx(G)
    encode_dim = 0
    trains, vals, tests = split(x_tensor, diagnosis_tensor, G_convert, encode_dim)
    return trains, vals, tests


def test_model(model_path=None):
    model = torch.load(model_path)
    model.eval()
    device = torch.device('cuda')
    trains, vals, tests = pre_data(device)
    test_loader = DataLoader(tests, batch_size=64)
    *rating, b_acc, b_model = evaluate(model, device, test_loader, 0, None)
    print(f'Test Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, Sensitivity: {rating[2]:.4f}, '
          f'Specificity: {rating[3]:.4f}, AUC: {rating[6]:.4f}')
    plot_graph(rating)



def main():
    # writer = SummaryWriter()
    writer = None
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    a = set_args()

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

    trains, vals, tests = pre_data(device)
    train_loader = DataLoader(trains, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(vals, batch_size=args['batch_size'])
    test_loader = DataLoader(tests, batch_size=args['batch_size'])
    model = ASAP(**model_params)
    model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses, ss_scores = [], []
    # Keep the best model
    b_acc = 0
    b_model = None
    print("Hello World!")
    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, device, train_loader, optimizer, loss_fn)
        losses.append(loss)
        scheduler.step()
        *rating, b_acc, b_model = evaluate(model, device, valid_loader, b_acc, b_model)
        if writer is not None:
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', rating[0], epoch)
            writer.add_scalar('Precision/train', rating[1], epoch)
            writer.add_scalar('Sensitivity/train', rating[2], epoch)
            writer.add_scalar('Specificity/train', rating[3], epoch)
            writer.add_scalar('AUC/test', rating[6], epoch)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, '
              f'Sensitivity: {rating[2]:.4f}, Specificity: {rating[3]:.4f}, AUC: {rating[6]:.4f}')
        ss_scores.append(rating[2])

    torch.save(model, f'models/final_model.pt')
    *rating, b_acc, b_model = evaluate(b_model, device, valid_loader, b_acc, b_model)
    print(f'Valid Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, Sensitivity: {rating[2]:.4f}, '
          f'Specificity: {rating[3]:.4f}, AUC: {rating[6]:.4f}')
    *rating, b_acc, b_model = evaluate(b_model, device, test_loader, b_acc, b_model)
    print(f'Test Accuracy: {rating[0]:.4f}, Precision: {rating[1]:.4f}, Sensitivity: {rating[2]:.4f}, '
          f'Specificity: {rating[3]:.4f}, AUC: {rating[6]:.4f}')
    torch.save(b_model, 'models/b_model.pth')
    plot_graph(losses, ss_scores, rating)


def plot_graph(rating, losses=None, ss_scores=None):
    if losses:
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label='Training Loss')
        plt.title('Loss during training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig('loss.png')

    if ss_scores:
        plt.figure(figsize=(12, 6))
        plt.plot(ss_scores, label='Sensitivity Score')
        plt.title('Sensitivity Score during training')
        plt.xlabel('Epoch')
        plt.ylabel('Sensitivity Score')
        plt.legend()
        plt.show()
        plt.savefig('ss_scores.png')

    plt.figure(figsize=(12, 6))
    plt.plot(rating[4], rating[5], color='darkorange', lw=2, label=f'ROC curve (AUC = {rating[6]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('fig.png')
    plt.show()


if __name__ == '__main__':
    # main()
    test_model('models/model_7878_7611.pt')
