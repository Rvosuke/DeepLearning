import torch
import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_curve, roc_auc_score


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        target = batch.y.float()
        target = target.reshape(-1, 2)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

    return loss.item()


def evaluate(model, device, loader, best_acc, best_model):
    model.eval()

    p_labels, t_labels, p_scores = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        output = torch.softmax(output, dim=1)
        target = data.y.float()

        p_labels.append(output.argmax(dim=1).cpu().numpy())
        p_scores.append(output[:, 1].cpu().numpy())
        target = target.reshape(-1, 2).argmax(dim=1)
        t_labels.append(target.cpu().numpy())

    p_label, t_label, p_score = np.concatenate(p_labels), np.concatenate(t_labels), np.concatenate(p_scores)

    accuracy = accuracy_score(t_label, p_label)
    precision = precision_score(t_label, p_label)
    tn, fp, fn, tp = np.ravel(np.array(confusion_matrix(t_label, p_label)))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    fpr, tpr, _ = roc_curve(t_label, p_score)
    auc = roc_auc_score(t_label, p_score)

    # Choose the best model according to accuracy
    if accuracy >= best_acc:
        best_acc = accuracy
        best_model = model
        print(f"Best model updated: accuracy = {best_acc:.4f}")

    # If both sensitivity and specificity are greater than 0.8, then save the model
    if sensitivity >= 0.75 and specificity >= 0.75:
        torch.save(model, f'models/model_{int(sensitivity * 1e4)}_{int(specificity * 1e4)}.pt')
        print("Model saved!")
        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'ROC_{int(sensitivity * 1e4)}_{int(specificity * 1e4)}.png')
        plt.show()

    return accuracy, precision, sensitivity, specificity, fpr, tpr, auc, best_acc, best_model
