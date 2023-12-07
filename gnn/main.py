import os
import copy
import torch
import torch_geometric

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from datas import data_split
from models import GCNGraph
from utils import train, eval


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

    if 'IS_GRADESCOPE_ENV' not in os.environ:
        evaluator = Evaluator(name='ogbg-molhiv')

        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_model = None
        best_valid_acc = 0

        for epoch in range(1, 1 + args["epochs"]):
            print('Training...')
            loss = train(model, device, train_loader, optimizer, loss_fn)

            print('Evaluating...')
            train_result = eval(model, device, train_loader, evaluator)
            val_result = eval(model, device, valid_loader, evaluator)
            test_result = eval(model, device, test_loader, evaluator)

            train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], \
                test_result[dataset.eval_metric]
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model = copy.deepcopy(model)
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
            return best_model, best_valid_acc


if __name__ == '__main__':
    main()
