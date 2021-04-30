from enum import IntEnum


class BlockType(IntEnum):
    Zero = 0
    String = 1
    Data = 2
    Pointer = 3

    def __str__(self):
        return self._name_
