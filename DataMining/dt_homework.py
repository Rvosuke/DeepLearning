import numpy as np
import matplotlib.pyplot as plt

train = np.array([
    [1, 1, 1, 1, 2, 2, 2, 2],
    [1, 1, 2, 2, 1, 1, 2, 2],
    [1, 2, 1, 2, 1, 2, 1, 2]
]).reshape(8, 3)
label = np.array([3, 1, 3, 2, 3, 2, 3, 3])


# 计算熵
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


# 计算信息增益
def information_gain(X, y, feature_index):
    initial_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_index], return_counts=True)
    probabilities = counts / len(y)
    weighted_entropy = np.sum(
        [prob * entropy(y[X[:, feature_index] == value]) for prob, value in zip(probabilities, values)])
    return initial_entropy - weighted_entropy


# 决策树递归构建
def build_decision_tree(X, y, max_depth, current_depth=0):
    if current_depth == max_depth or len(np.unique(y)) == 1:
        return {"class": np.argmax(np.bincount(y))}

    # 选择具有最大信息增益的特征
    best_feature_index = np.argmax([information_gain(X, y, i) for i in range(X.shape[1])])

    # 按照最佳特征的值进行分割
    unique_values = np.unique(X[:, best_feature_index])
    children = {}
    for value in unique_values:
        child_X = X[X[:, best_feature_index] == value]
        child_y = y[X[:, best_feature_index] == value]
        children[value] = build_decision_tree(child_X, child_y, max_depth, current_depth + 1)

    return {"feature": best_feature_index, "children": children}


def plot_tree(decision_tree, ax=None):
    if ax is None:
        ax = plt.gca()
    if "class" in decision_tree:
        ax.text(0.5, 0.5, f"class={decision_tree['class']}", ha="center", va="center")
        return
    feature_index = decision_tree["feature"]
    children = decision_tree["children"]
    for value, child_tree in children.items():
        ax.text(0.5, 0.5, f"X[{feature_index}]={value}", ha="center", va="center")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False))
        ax.plot([0.5, 0.5], [0, 1], "--", color="gray")
        ax.plot([0, 0.5], [0.5, 0.5], "--", color="gray")
        ax.plot([0.5, 1], [0.5, 0.5], "--", color="gray")
        ax.text(0.25, 0.25, "yes", ha="center", va="center")
        ax.text(0.75, 0.25, "no", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plot_tree(child_tree, ax)


# 构建决策树
decision_tree = build_decision_tree(train, label, max_depth=4)
fig, ax = plt.subplots(figsize=(5, 5))
plot_tree(decision_tree, ax)
plt.show()
# print("决策树:", decision_tree)
