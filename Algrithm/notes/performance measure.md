# Performance Measure

## Classification
### error rate and accuracy
错误率是指分类器预测错误的样本数占总样本数的比例，准确率是指分类器预测正确的样本数占总样本数的比例。
一般来说，错误率越小，分类器的性能越好，准确率越大，分类器的性能越好。
具体计算公式见下：
错误率（error） = 错误分类的样本数 / 总样本数
准确率（accuracy） = 正确分类的样本数 / 总样本数

### precision and recall
查准率和召回率是衡量分类器性能的两个重要指标。
查准率是指分类器预测为正的样本中有多少是真正的正样本，召回率是指所有真正的正样本中有多少被分类器预测为正。
一般来说，查准率越大，分类器的性能越好，召回率越大，分类器的性能越好。
具体计算公式见下：
查准率（precision） = 正确分类的正样本数 / 分类器预测为正的样本数
也写作 
$$
P = TP / (TP + FP)
$$
召回率（recall） = 正确分类的正样本数 / 所有真正的正样本数
也写作 
$$
R = TP / (TP + FN)
$$


### F1 score
F1 score是准确率和召回率的调和平均值，一般来说，F1 score越大，分类器的性能越好。
具体计算公式见下：
F1 score = 2 * 准确率 * 召回率 / (准确率 + 召回率)
即 
$$
F_1 = 2 * P * R / (P + R)
$$
F1 score的一般形式：
Fβ score = (1 + β^2) * 准确率 * 召回率 / (β^2 * 准确率 + 召回率)
即 
$$
F_β = (1 + β^2) * P * R / (β^2 * P + R)
$$
β = 1时，Fβ score = F1 score
β > 1时，Fβ score > F1 score，此时查全率有更大的影响
β < 1时，Fβ score < F1 score，此时查准率有更大的影响

### PR curve

PR曲线是一种常用的二分类模型性能评估方法，PR曲线的横坐标是召回率（recall），纵坐标是准确率（precision）。
PR曲线下的面积（AP）是常用的衡量二分类模型性能的指标，AP越大，模型性能越好。
具体计算公式见下：
$$
AP = \frac{\sum(Precision_i)} {n} 
$$
其中，
$$
Precision_i = TP / (TP + FP)
$$


### ROC curve and AUC

ROC曲线是一种常用的二分类模型性能评估方法，ROC曲线的横坐标是假正率（FPR），纵坐标是真正率（TPR）。
ROC曲线下的面积（AUC）是常用的衡量二分类模型性能的指标，AUC越大，模型性能越好。
具体计算公式见下：
$$
FPR = FP / (FP + TN)
$$

$$
TPR = TP / (TP + FN)
$$

$$
AUC = 1 / 2 * (TPR + TNR) * (FPR - TNR)
$$

$$
TNR = \frac{TN}{TN + FP}
$$

其中，TPR = recall，FPR = 1 - TNR

## Regression

### Mean Absolute Error (MAE)
MAE是回归问题中常用的评价指标，MAE越小，回归模型的性能越好。
具体计算公式见下：
$$
MAE = \frac{\sum(|y_i - \hat{y_i}|)}{n} 
$$


### Mean Squared Error (MSE)

MSE是回归问题中常用的评价指标，MSE越小，回归模型的性能越好。
具体计算公式见下：
$$
MSE = 1 / n * \sum(y_i - \hat{y_i})^2
$$


### Root Mean Squared Error (RMSE)

RMSE是回归问题中常用的评价指标，RMSE越小，回归模型的性能越好。
具体计算公式见下：
$$
RMSE = \sqrt{\frac{\sum(y_i - \hat{y_i})^2} {n}}
$$


### Mean Absolute Percentage Error (MAPE)

MAPE是回归问题中常用的评价指标，MAPE越小，回归模型的性能越好。
具体计算公式见下：
$$
MAPE = \frac{\sum|(y_i - \hat{y_i}) / y_i|}{n}
$$
