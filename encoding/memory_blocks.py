import mmap
from abc import ABC, abstractmethod
from encoding import BlockType
from typing import List
from string import printable


def _determine_byte_type(byte: int) -> BlockType:

    if byte == 0:
        return BlockType.Zero
    elif chr(byte) in printable:
        return BlockType.String
    else:
        return BlockType.Data


class MemoryEncoder(ABC):
    def __init__(self, file_path: str, pointer_size: int):
        self.file_path = file_path
        self.pointer_size: int = pointer_size

    @property
    def mmap(self):
        with open(self.file_path, "rb") as f:
            map = mmap.mmap(f.fileno(), f.seek(0, 2), access=mmap.ACCESS_READ)
        return map

    @abstractmethod
    def encode(self, file, offset: int) -> List[BlockType]:
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


class BallEncoder(MemoryEncoder):
    """
    Encodes the immediate surroundings of given offset. (Usually the offset of a pointer)
    """

    def __init__(self, *args, radius: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius

    def encode(self, offset):
        raise NotImplementedError
