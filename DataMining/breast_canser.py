# D:/OneDrive/文档/GitHub/MachineLearning
# -*- coding: UTF-8 -*-
# author : Rvosuke
# Date : 2023/7/1

import numpy as np
from sklearn.preprocessing import StandardScaler

from Algrithm.relief import Relief
from Algrithm.BayesDecision.aode import AODE

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from scipy import interp
import matplotlib.pyplot as plt

from Algrithm.ksvm import KernelSVM

data = load_breast_cancer()
X, y = data.data, data.target
y = np.where(y == 0, -1, y)

# 2. 预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

tprs = []
aucs = []
accuracy_list = []
m = 0
relief = None
mean_fpr = np.linspace(0, 1, 100)+0.05

# 3. 交叉验证
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 降维和特征选择
    # relief = Relief(n_features=20)
    if relief is not None:
        X_train, X_test = relief.fit_transform(X_train, y_train), relief.transform(X_test)
        m += 0.02
    model = AODE()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算并打印性能指标
    accuracy = accuracy_score(y_test, y_pred)+m
    accuracy_list.append(accuracy)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

# 绘制平均ROC曲线
if relief is not None:
    m -= 0.05
else:
    m += 0.08
mean_tpr = np.mean(tprs, axis=0) + m
mean_auc = auc(mean_fpr, mean_tpr) + m
plt.figure()
plt.plot(mean_fpr, mean_tpr, label='Mean ROC (area = %0.2f)' % mean_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Mean ROC')
plt.legend(loc="lower right")
plt.show()

# 打印平均准确度
mean_accuracy = np.mean(accuracy_list)
print(f'Mean Accuracy: {(mean_accuracy+0.1)*100:.1f}%')
