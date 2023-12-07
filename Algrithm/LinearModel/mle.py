"""I'm using the maximum likelihood estimation（MLE） method to estimate the class conditional probabilities of the
first three attributes in the watermelon dataset 3.0. My first assumption is that these discrete sample attributes
follow a multinomial distribution. Therefore, all I need to do is calculate the probability of each attribute's
different values under each category. This is the maximum likelihood estimation. """
import numpy as np
from collections import Counter

# 需求不同，这里采取稍微不同的打开方式
with open('../../res/data_3.0.txt', mode='r', encoding='utf-8') as f:
    attributes_labels = f.readline().strip().split(',')
    # 只读取前三个属性
    attributes_labels = attributes_labels[1: 4] + attributes_labels[-1:]  # 选取前三个属性和最后一个类别
    text = []
    for line in f.readlines():
        line = line.strip().split(',')
        line = line[1: 4] + line[-1:]  # 选取前三个属性和最后一个类别
        text.append(line)
    f.close()


class_count = Counter()
for i in range(len(text)):
    val = text[i][-1]
    class_count[val] += 1
class_len = len(class_count)
class_name = list(class_count.keys())  # 存储类别名称


attributes_len = 3
# condition_p = np.zeros((class_len, attributes_len), dtype=float)
condition_p_list = []
# 计算类条件概率
for i in range(attributes_len):
    condition_p = []
    for j in range(class_len):
        attribute_values = [row[i] for row in text if row[-1] == class_name[j]]
        attribute_count = Counter(attribute_values)
        attribute_p = [attribute_count[val] / class_count[class_name[j]] for val in list(attribute_count.keys())]
        condition_p.append(attribute_p)
    condition_p_list.append(condition_p)

print('类别名称：', class_name)
print('类条件概率：', condition_p_list)
# exit(0)
