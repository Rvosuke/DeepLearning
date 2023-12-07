import torch
import copy
import pandas as pd


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        # ignore NaN targets (unlabeled) when computing training loss.
        is_labeled = y == y
        loss = loss_fn(out[is_labeled], y[is_labeled])
        loss.backward()
        optimizer.step()

    return loss.item()


def eval(model, device, loader, evaluator, save_model_results=False):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if save_model_results:
        print("Saving Model Predictions")

        # Create a pandas dataframe with a two columns
        # y_pred | y_true
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)

        df = pd.DataFrame(data=data)
        df.to_csv('graph_evaluate.csv')

    return evaluator.eval(input_dict)

