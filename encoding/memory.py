import mmap
from abc import ABC, abstractmethod
from encoding import BlockType
from typing import List
from string import printable


def _determine_byte_type(byte: int) -> BlockType:
    if byte == "\0":
        return BlockType.Zero
    elif chr(byte) in printable:
        return BlockType.String
    else:
        return BlockType.Data


class MemoryEncoder(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

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
    Encodes memory between "start and end" pointers.
    """

    def __init__(self, *args, max_size=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def encode(self, start, end):
        # TODO: What about start == 0?
        if start + 1 == end:
            return [BlockType.Pointer, BlockType.Pointer]
        else:
            encoded = [_determine_byte_type(char) for char in self.mmap[start + 1 : end]]
            return [BlockType.Pointer] + encoded + [BlockType.Pointer]


class BallEncoder(MemoryEncoder):
    """
    Encodes the immediate surroundings of given offset. (Usually the offset of a pointer)
    """

    def __init__(self, *args, radius: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius

    def encode(self, offset):
        raise NotImplementedError
