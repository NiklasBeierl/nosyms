import mmap
from abc import ABC, abstractmethod
from functools import cached_property
import numpy as np
from nosyms.encoding import BlockType


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
    def as_numpy(self) -> np.array:
        raw = np.array(self.mmap, dtype=np.int8)
        result = np.full(len(raw), BlockType.Unknown.value, dtype=np.int8)
        printable = ((9 <= raw) & (raw <= 13)) | ((32 <= raw) & (raw <= 126))  # Printable ascii
        result[printable] = BlockType.String.value
        return result

    @abstractmethod
    def encode(self, *args, **kwargs) -> np.array:
        ...
