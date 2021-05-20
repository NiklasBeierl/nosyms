import mmap
from abc import ABC, abstractmethod
from typing import List
from string import printable
import torch as t
import numpy as np
from encoding import BlockType


def _determine_byte_type(byte: int) -> BlockType:
    if chr(byte) in printable:
        return BlockType.String
    else:
        return BlockType.Data


def _determine_byte_type_int(byte: int) -> int:
    if chr(byte) in printable:
        return BlockType.String.value
    else:
        return BlockType.Data.value


class MemoryEncoder(ABC):
    def __init__(self, file_path: str, pointer_size: int):
        self.file_path = file_path
        self.pointer_size: int = pointer_size
        self._as_numpy = None
        self._mmap = None

    @property
    def mmap(self):
        if self._mmap is None:
            with open(self.file_path, "rb") as f:
                self._mmap = mmap.mmap(f.fileno(), f.seek(0, 2), access=mmap.ACCESS_READ)
        return self._mmap

    @property
    def as_numpy(self) -> t.tensor:
        if self._as_numpy is None:
            self._as_numpy = np.zeros(len(self.mmap), dtype=t.int8)
            # TODO: There is probably a faster way to do that.
            for i, b in enumerate(self.mmap):
                self._as_numpy[i] = _determine_byte_type_int(b)
        return self._as_numpy

    @abstractmethod
    def encode(self, *args, **kwargs) -> List[BlockType]:
        ...


class SandwichEncoder(MemoryEncoder):
    """
    Encodes memory between "start and end". Assumes that "start" points to the first and "end"
    to the last byte of a pointer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bread = [BlockType.Pointer] * self.pointer_size

    def encode(self, start, end):
        # TODO: What about start == 0?
        if start + self.pointer_size == end:
            return [BlockType.Pointer, BlockType.Pointer] * self.pointer_size
        else:
            encoded = [
                _determine_byte_type(char)
                for char in self.mmap[start + self.pointer_size : end - self.pointer_size + 1]
            ]
            return self._bread + encoded + self._bread
