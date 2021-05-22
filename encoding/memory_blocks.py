import mmap
from abc import ABC, abstractmethod
from string import printable
from functools import cached_property
import torch as t
import numpy as np
from encoding import BlockType

PRINTABLE_BYTES = printable.encode("ascii")


def _determine_byte_type(byte: int) -> BlockType:
    if chr(byte) in printable:
        return BlockType.String
    else:
        return BlockType.Data


def _determine_byte_type_int(byte: bytes) -> int:
    if byte in PRINTABLE_BYTES:
        return BlockType.String.value
    else:
        return BlockType.Data.value


class MemoryEncoder(ABC):
    def __init__(self, file_path: str, pointer_size: int):
        self.file_path = file_path
        self.pointer_size: int = pointer_size

    @cached_property
    def size(self) -> int:
        return self.mmap.size()

    @cached_property
    def mmap(self):
        with open(self.file_path, "rb") as f:
            return mmap.mmap(f.fileno(), f.seek(0, 2), access=mmap.ACCESS_READ)

    @cached_property
    def as_numpy(self) -> t.tensor:
        as_numpy = np.zeros(self.size, dtype=np.int8)
        # TODO: There is probably a faster way to do that.
        for i, b in enumerate(self.mmap):
            as_numpy[i] = _determine_byte_type_int(b)
        return as_numpy

    @abstractmethod
    def encode(self, *args, **kwargs) -> t.tensor:
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
