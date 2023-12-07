import torch
import copy
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, confusion_matrix


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        # ignore NaN targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        loss = loss_fn(out[is_labeled], batch.y[is_labeled])
        loss.backward()
        optimizer.step()

    return loss.item()


def eval(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)

            predicted_labels = (output > 0.5).float()

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    tn, fp, fn, tp = np.ravel(np.array(confusion_matrix(y_true, y_pred)))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, precision, sensitivity, specificity, fpr, tpr, auc
