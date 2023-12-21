import torch
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, confusion_matrix


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        target = batch.y.float()
        target = target.reshape(-1, 2)
        # print(out.shape)
        # ignore NaN targets (unlabeled) when computing training loss.
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate(model, device, loader, best_acc, best_model):
    model.eval()
    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)

        # predicted_labels = (output > 0.5).float()
        predicted_labels = output.argmax(dim=1)
        target = data.y.float()
        target = target.reshape(-1, 2)
        target = target.argmax(dim=1)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    tn, fp, fn, tp = np.ravel(np.array(confusion_matrix(y_true, y_pred)))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    # Choose the best model according to accuracy
    if accuracy > best_acc:
        best_acc = accuracy
        best_model = model

    return accuracy, precision, sensitivity, specificity, fpr, tpr, auc, best_acc, best_model
