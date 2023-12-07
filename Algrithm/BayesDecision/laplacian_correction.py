import numpy as np
from scipy.stats import stats

from navie_bayes import text, attributes_len, is_float, attributes_count_list, guess_means, guess_stds, guess_index, \
    class_name

test = ['青绿', '蜷缩', '清脆', '清晰', '凹陷', '硬滑', '0.697', '0.460', '?']
# 计算修正后的先验概率，没有使用Counter，效率应该会更高一点
class_count = {}
for i in range(len(text)):
    val = text[i][-1]
    if val not in class_count:
        class_count[val] = 0
    class_count[val] += 1

class_len = len(class_count)
prior_p = np.zeros(class_len, dtype=float)
for i in range(class_len):
    prior_p[i] = (class_count[i] + 1) / (len(text) + class_len)

print(prior_p)
# 接下来计算样本属性的各个类条件概率
condition_p = np.zeros((class_len, attributes_len), dtype=float)
for k in range(class_len):
    for i, val in enumerate(test[:-1]):
        if not is_float(val):
            if val not in attributes_count_list[k]:
                attributes_count_list[k][i][val] += 1
            class_condition_p = [
                (attributes_count_list[k][i][val] + 1) / (class_count[k] + len(attributes_count_list[k][i]))
                for k in range(class_len)]
        else:
            j = list(guess_index).index(i)
            class_condition_p = [stats.norm.pdf(float(val), guess_means[k][j], guess_stds[k][j])
                                 for k in range(class_len)]
            # class_condition_p = [1 / (uniform_max[k][j] + uniform_min[k][j]) for k in range(class_len)]  # 均匀分布估计
        condition_p[:, i][:] = class_condition_p

# 接着进行判别函数的计算，这里是朴素贝叶斯，条件独立性假设
class_g = np.zeros(class_len, dtype=float)
for i in range(class_len):
    class_g[i] = prior_p[i] * np.prod(condition_p[i])

# 根据判别函数做出决策
decision = class_name[np.argmax(class_g)]
print(decision)
exit(0)
