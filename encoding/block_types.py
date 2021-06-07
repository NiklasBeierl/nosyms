from enum import IntEnum
from abc import ABC, abstractmethod
from typing import List
import torch as t
import numpy as np


class BlockType(IntEnum):
    Unknown = 0
    Data = 1
    Pointer = 2
    String = 3

    def __str__(self):
        return self._name_


def blocks_to_numpy(blocks: List[BlockType]) -> np.array:
    """
    Turn a list of BlockTypes  into a numpy array where tensor[i] == list[i].value
    :param blocks: List of BlockTypes to encode.
    :return: numpy array of encoded BlockTypes.
    """
    return np.array([b.value for b in blocks], dtype=np.int)


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
        # Mode chooses the lower value in case of a tie
        result, _ = batch.mode(dim=2)

        # Deal with remaining blocks
        if height % self.word_size != 0:
            leftovers = blocks[:, batch_height:]
            leftovers, _ = leftovers.mode()
            result = t.hstack((result, leftovers.reshape(-1, 1)))
        return result
