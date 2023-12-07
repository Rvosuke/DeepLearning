import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 导入数据集
california = fetch_california_housing()
X = california.data
y = california.target.reshape(-1, 1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型并进行训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)
print('均方误差：', mean_squared_error(y_test, y_pred))
print('R2得分：', r2_score(y_test, y_pred))
