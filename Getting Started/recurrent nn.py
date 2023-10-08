import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def get_params(vocab_size, num_hiddens, divice):
    num_inputs = num_outputs = vocab_size

    def normal(shape: tuple):
        return torch.randn(size=shape, device=divice) * 0.01

    Wxh = normal((num_inputs, num_hiddens))
    Whh = normal((num_hiddens, num_hiddens))
    bh = normal(num_hiddens)

    Whp = normal((num_hiddens, num_outputs))
    bp = normal(num_outputs)

    params = Wxh, Whh, bh, Whp, bp

    for param in params:
        param.requires_grad_(True)
    return params
