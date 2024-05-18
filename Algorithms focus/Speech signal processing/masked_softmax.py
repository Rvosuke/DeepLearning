import torch
from torch import nn
# from d2l import torch as d2l


def mask_softmax(X: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
    """
    Perform softmax operation by masking elements on the last axis.
    :param X: 3D (Tensor) of shape (batch_size, seq_len, vocabulary_size)
    :param valid_lens: 1D or 2D (Tensor) of valid lengths of sequences
    :return: 3D (Tensor) of shape (batch_size, seq_len, vocabulary_size)
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=shape[1], dim=0)
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


if __name__ == '__main__':
    out = mask_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
    print(out)
