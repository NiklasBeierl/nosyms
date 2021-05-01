from enum import IntEnum
from typing import List
import torch as t


class BlockType(IntEnum):
    Zero = 0
    String = 1
    Data = 2
    Pointer = 3

    def __str__(self):
        return self._name_


def blocks_to_tensor_truncate(blocks: List[BlockType], tensor_length: int = 100) -> t.tensor:
    """
    Convert list of BlockTypes to tensor of shape (tensor_length, len(BlockType)). Every row in the tensor "one-hot"
    encodes the corresponding block in `blocks`. If `blocks` is longer than `tensor_length`, it is truncated.
    If it is shorter, the the remaining "rows" will be all 0.
    :param blocks: List of blocks to be encoded.
    :param tensor_length: Length of the resulting tensor.
    :return: tensor of shape (tensor_length, len(BlockType))
    """
    output = t.zeros(tensor_length, len(BlockType))
    for i, b in enumerate(blocks[:tensor_length]):
        output[i][b] = 1
    return output