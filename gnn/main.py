import warnings
import torch

from torch_geometric.loader import DataLoader

from datas import data_split
from models import GCNGraph
from utils import train, evaluate

warnings.filterwarnings('ignore')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = {
        'batch_size': 32,
        'gcn_params': {
            'in_channels': 4,
            'hidden_dim': 256,
            'out_channels': 1,
            'gcn_layerss': 5,
            'gcn_base_layers': 3,
            'dropout': 0.3,
        }
        'lr': 1e-3,
        'epochs': 100,
    }
    train_loader = DataLoader(data_split[0], batch_size=args['batch_size'], shuffle=True)
    valid_loader, test_loader = DataLoader(data_split[1]), DataLoader(data_split[2])
    model = GCNGraph(**args[gcn_params]).to(device)
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
