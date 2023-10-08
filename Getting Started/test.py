import random
import torch
import torch.nn.functional as F

X = torch.arange(0, 10)
X.reshape(2, 5)

X = F.one_hot(X.T, 28)
print(X)
