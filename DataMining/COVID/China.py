# %%
import pandas as pd

# Read the data
df = pd.read_csv('DXYArea.csv')

# 选中含有‘中国’的行
df = df.loc[df['province_zipCode'] == 951001]

# save the data
df.to_csv('covid_china.csv', index=False, encoding='utf-8-sig')

# %% [数据处理]
# ## 现在我们将数据集中的最后一列‘updataTime’转化为时间戳，并利用time模块来将时间转化为从某个时间点开始的时间差，单位为分钟

# 转化最后一列为时间戳
df['updateTime'] = pd.to_datetime(df['updateTime'])

# 记录从最早的时间点开始的时间差
df['updateTimeM'] = df['updateTime'].apply(lambda x: (x - df['updateTime'].min()).total_seconds() / 60).astype(int)
df['updataTimeD'] = df['updateTime'].apply(lambda x: (x - df['updateTime'].min()).total_seconds() / 86400).astype(int)

# 打印数据集
df

# %% [预测确诊人数]
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# 在进行模型拟合之前，我们需要观察数据，判断是否需要进行数据处理。
plt.figure(figsize=(10, 6))
plt.scatter(df['updateTimeM'], df['province_deadCount'])
plt.xlabel('updateTime')
plt.ylabel('province_deadCount')
plt.show()

# 通过观察发现按分钟隔开效果更好，
# 接着我们选取分钟作为特征，‘province_confirmedCount’作为标签，将数据集分为训练集和测试集
X_data = df['updateTimeM'].values.reshape(-1, 1)
Y_data = df['province_confirmedCount'].values.reshape(-1, 1)

# 构建线性模型
rig = Ridge(alpha=0.1)

# 使用交叉验证划分训练集和测试集
mse1 = cross_val_score(rig, X_data, Y_data, scoring='neg_mean_squared_error', cv=5)

# 取平均值
print('Train mae:', np.sqrt(-np.mean(mse1)))

# %% [预测治愈人数]
# 基本原理与上述相同，先经过特征工程，再进行模型训练
X_data = df['updateTimeM']
Y_data = df['province_curedCount']
X_data = X_data.values.reshape(-1, 1)
Y_data = Y_data.values.reshape(-1, 1)
#%% [绘制'updateTimeM'与'province_curedCount'的散点图]
plt.figure(figsize=(10, 6))
plt.scatter(df['updateTimeM'], df['province_curedCount'])
plt.xlabel('updateTime')
plt.ylabel('province_curedCount')
plt.show()
#%% [训练模型]

# 构建线性模型
rig = Ridge(alpha=10)

# 使用交叉验证划分训练集和测试集
kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 5折交叉验证
scores_train = []
scores = []
for train_ind, val_ind in kf.split(X_data, Y_data):
    train_x = X_data[train_ind]
    train_y = Y_data[train_ind]
    val_x = X_data[val_ind]
    val_y = Y_data[val_ind]

    rig.fit(train_x, train_y)
    pred_train_xgb = rig.predict(train_x)
    pred_xgb = rig.predict(val_x)

    score_train = mean_squared_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = mean_squared_error(val_y, pred_xgb)
    scores.append(score)

# 取平均值
print('Train mae:', np.mean(scores_train))
print('Val mae', np.mean(scores))
# %% 绘制拟合图形
plt.figure(figsize=(10, 6))
plt.scatter(df['updateTimeM'], df['province_curedCount'])
plt.plot(df['updateTimeM'], rig.predict(X_data), color='red')
plt.xlabel('updateTime')
plt.ylabel('province_curedCount')
plt.show()


# %% [预测死亡人数]
# 接着我们选取最后一列作为特征，‘province_confirmedCount’作为标签，将数据集分为训练集和测试集
X_data = df['updateTimeM']
Y_data = df['province_deadCount']
X_data = X_data.values.reshape(-1, 1)
Y_data = Y_data.values.reshape(-1, 1)

# 构建线性模型
rig = Ridge(alpha=0.1)

# 使用交叉验证划分训练集和测试集
kf = KFold(n_splits=5, shuffle=True, random_state=0)  # 5折交叉验证
scores_train = []
scores = []
for train_ind, val_ind in kf.split(X_data, Y_data):
    train_x = X_data[train_ind]
    train_y = Y_data[train_ind]
    val_x = X_data[val_ind]
    val_y = Y_data[val_ind]

    rig.fit(train_x, train_y)
    pred_train_xgb = rig.predict(train_x)
    pred_xgb = rig.predict(val_x)

    score_train = mean_squared_error(train_y, pred_train_xgb)
    scores_train.append(score_train)
    score = mean_squared_error(val_y, pred_xgb)
    scores.append(score)

# 取平均值
print('Train mae:', np.mean(scores_train))
print('Val mae', np.mean(scores))
