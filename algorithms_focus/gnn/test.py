import torch
import torch.nn.functional as F

# 设置随机种子
torch.manual_seed(0)
# 生成一个包含10个元素的随机整数Tensor，每个元素的值在0到2之间
labels = torch.randint(0, 3, (10,))

# 将标签Tensor转换为one-hot编码
one_hot_labels = F.one_hot(labels, num_classes=3)

print("原始标签Tensor：", labels)
print("One-hot编码后的标签Tensor：", one_hot_labels)