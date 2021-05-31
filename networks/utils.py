import torch as t
from torch.nn.functional import one_hot


def one_hot_with_neutral(tensor: t.tensor, neutral_class: int = 0, **kwargs):
    result = one_hot(tensor, **kwargs)
    classes = result.shape[-1]
    if neutral_class >= classes:
        raise ValueError("Neutral class not in classes.")
    result = result[:, :, [i for i in range(result.shape[-1]) if i != neutral_class]]
    return result
