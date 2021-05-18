from enum import IntEnum
from typing import List
import torch as t
from hyperparams import NODE_MAX_LEN


class BlockType(IntEnum):
    String = 3
    Data = 1
    Zero = 1  # I discovered that I basically never get zero where I expect it.
    Pointer = 2

    def __str__(self):
        return self._name_


def blocks_to_tensor_truncate(blocks: List[BlockType], tensor_length: int = NODE_MAX_LEN) -> t.tensor:
    """
    Convert list of BlockTypes to tensor of shape (tensor_length, len(BlockType)). Every row in the tensor "one-hot"
    encodes the corresponding block in `blocks`. If `blocks` is longer than `tensor_length`, it is truncated.
    If it is shorter, the the remaining "rows" will be all 0.
    :param blocks: List of blocks to be encoded.
    :param tensor_length: Length of the resulting tensor.
    :return: tensor of shape (tensor_length, len(BlockType))
    """
    num_ts = len(BlockType)
    output = t.zeros(tensor_length * len(BlockType))
    for i, b in enumerate(blocks[:tensor_length]):
        output[i * num_ts + b] = 1
    return output.to_sparse()
