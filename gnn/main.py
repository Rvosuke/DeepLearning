import warnings
import torch
import torch_geometric

from datas import data_split
from models import GCNGraph
from utils import train, evaluate

warnings.filterwarnings('ignore')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = {
        'batch_size': 32,
        'hidden_dim': 256,
        'num_layers': 5,
        'dropout': 0.3,
        'lr': 1e-3,
        'epochs': 50,
    }
    train_loader = torch_geometric.loader.DataLoader(data_split[0], batch_size=args['batch_size'], shuffle=True)
    valid_loader = torch_geometric.loader.DataLoader(data_split[1])
    test_loader = torch_geometric.loader.DataLoader(data_split[2])
    model = GCNGraph(4, args['hidden_dim'], 1, args['num_layers'], args['dropout']).to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # Keep the best model
    b_acc = 0
    b_model = None
    for epoch in range(1, 1 + args["epochs"]):
        loss = train(model, device, train_loader, optimizer, loss_fn)
        accuracy, precision, sensitivity, specificity, _, _, auc, b_acc, b_model \
            = evaluate(model, device, valid_loader, b_acc, b_model)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
              f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, AUC: {auc:.4f}')

    accuracy, precision, sensitivity, specificity, fpr, tpr, auc, _, _ \
        = evaluate(b_model, device, test_loader, b_acc, b_model)
    print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Sensitivity: {sensitivity:.4f}, '
          f'Specificity: {specificity:.4f}, fpr: {fpr}, tpr: {tpr}, AUC: {auc:.4f}')


if __name__ == '__main__':
    main()
