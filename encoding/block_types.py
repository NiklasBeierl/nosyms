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


_BLOCKS_TO_INT = {bt: bt.value for bt in BlockType}
_BLOCKS_TO_INT[None] = 0


def blocks_to_tensor(blocks: List[BlockType]) -> t.tensor:
    """
    Turn a list of BlockTypes and `None` int a tensor where tensor[i] == list[i].value
    and tensor[i] == 0 if list[i] == None.
    :param blocks: List of BlockTypes to encode.
    :return: Tensor of encoded BlockTypes.
    """
    # TODO: This probably has room for optimization (np.vectorize?).
    result = t.zeros(len(blocks), dtype=t.int8)
    for i, block in enumerate(blocks):
        result[i] = _BLOCKS_TO_INT[block]
    return result
