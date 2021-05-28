from enum import IntEnum
from abc import ABC, abstractmethod
from typing import List
import torch as t
import numpy as np


class BlockType(IntEnum):
    String = 3
    Data = 1
    Zero = 1  # I discovered that I basically never get zero where I expect it.
    Pointer = 2

    def __str__(self):
        return self._name_


_BLOCKS_TO_INT = {bt: bt.value for bt in BlockType}
_BLOCKS_TO_INT[None] = 0


def blocks_to_numpy(blocks: List[BlockType]) -> np.array:
    """
    Turn a list of BlockTypes and `None` int a numpy array where tensor[i] == list[i].value
    and tensor[i] == 0 if list[i] == None.
    :param blocks: List of BlockTypes to encode.
    :return: numpy array of encoded BlockTypes.
    """
    # TODO: This probably has room for optimization (np.vectorize?).
    result = np.zeros(len(blocks), dtype=np.int)
    for i, block in enumerate(blocks):
        result[i] = _BLOCKS_TO_INT[block]
    return result


class BlockCompressor(ABC):
    @abstractmethod
    def compress(self, blocks: "Tensor: a") -> "Tensor: b":
        """
        Compresses a memory chunks "blocks" vector from length "a" to length "b"
        :param blocks: 1 dim tensor representing a memory chunks "blocks"
        :return: 1 dim tensor representing the compressed "blocks"
        """
        ...

    # We could also do one method, but explicit is better than implicit
    @abstractmethod
    def compress_batch(self, blocks: "Tensor: n x a") -> "Tensor: n x b":
        """
        Batch version of "compress"
        :param blocks: Tensor: n x a holding the "blocks" vectors of n chunks having length a
        :return: Tensor: n x b holding the compressed "blocks"
        """
        ...


class WordCompressor(BlockCompressor):
    def __init__(self, word_size=8):
        self.word_size = word_size

    def compress(self, blocks: "Tensor: a") -> "Tensor: b":
        return self.compress_batch(blocks.reshape(1, -1))

    def compress_batch(self, blocks: "Tensor: n x a") -> "Tensor: n x b":
        length, height = blocks.shape
        # If the chunk length is not a multiple of word_size we need to slice them out separately
        batch_height = height - (height % self.word_size)
        batch = blocks[:, :batch_height]
        batch = batch.reshape(length, batch_height // self.word_size, self.word_size)
        result, _ = batch.mode(dim=2)

        # Deal with remaining blocks
        if height % self.word_size != 0:
            leftovers = blocks[:, batch_height:]
            leftovers, _ = leftovers.mode()
            result = t.hstack((result, leftovers.reshape(-1, 1)))
        return result
