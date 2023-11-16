# Author: Rvosuke
# Date: 6-11-2023
# 梯度下降

import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
Y = np.array([2 * (x - 2) + 1 for x in X], dtype=np.float32)


# 初始化参数, 使用numpy初始化时候要注意调用的API是np.random.normal()而不是np.random.randn()
np.random.seed(0)
w, b = np.random.normal(), np.random.normal()


def f(x):
    """
    线性回归模型, 这里x完全可以是向量
    :param x:
    :return:
    """
    return w * x + b


def dw(x, y):
    """
    求目标函数对w的偏导, 这里我们完全没有给出目标函数, 表示在随机梯度下降中, 我们不需要知道目标函数的值
    :param x:
    :param y:
    :return:
    """
    return (f(x) - y) * x


def db(x, y):
    return f(x) - y


lr = 0.001

for i in range(2000):
    for x, y in zip(X, Y):
        w -= lr * dw(x, y)
        b -= lr * db(x, y)
print(f"w:{w:.2f}, b:{b:.2f}")

n = 4
X1, Y1 = X[:n], Y[:n]
X2, Y2 = X[n:2*n], Y[n:2*n]
X3, Y3 = X[2*n:3*n], Y[2*n:3*n]
X, Y = [X1, X2, X3], [Y1, Y2, Y3]
# 似乎在训练中, 每一次迭代都要遍历所有的数据, 这样的话, 似乎就不是随机梯度下降了
for i in range(2000):
    # 这里只是批量梯度下降, 并非随机梯度下降, 因为每一次迭代都要遍历所有的数据
    for x, y in zip(X, Y):
        dw_v, db_v = dw(x, y), db(x, y)
        dw_v = dw_v.mean()
        db_v = db_v.mean()
        w = w - lr * dw_v
        b = b - lr * db_v
print(w, b)


# 其他的梯度下降方法包括随机梯度下降, 即每次迭代只使用一个样本来更新参数, 但是这样的话, 会导致参数更新的不稳定, 甚至会导致参数不收敛
# 另外最常用的是小批量梯度下降, 即每次迭代使用一小部分样本来更新参数, 这样的话, 既可以保证参数的稳定性, 又可以保证参数的收敛性
# 这两种梯度下降方法的实现与上面的代码类似, 只是不在每一个迭代中遍历所有的样本, 而是遍历一个随机选择的小批量的样本
