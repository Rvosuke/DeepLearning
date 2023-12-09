import warnings
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from datas import split, read_data
from models import GCNGraph
from utils import train, evaluate

warnings.filterwarnings('ignore')


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
    args = {
        'batch_size': 32,
        'no_pos': True,  # if False, positional encoding will not be used
        'lr': 1e-3,
        'epochs': 100,
    }
    gcn_params = {
        'in_channels': 1 if args['no_pos'] else 4,
        'hidden_dim': 256,
        'out_channels': 1,
        'gcn_layers': 5,
        'gcn_base_layers': 3,
        'dropout': 0.3,
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
