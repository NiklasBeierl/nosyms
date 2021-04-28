from enum import IntEnum


def _singleton(cls):
    return cls()


class AtomicType:
    @_singleton
    class Padding:
        pass

    @_singleton
    class String:
        pass

    @_singleton
    class Data:
        pass

    class Pointer:
        def __init__(self, name: str):
            self.name: str = name

        def __repr__(self):
            return f"Pointer to: {self.name}"


class BlockType(IntEnum):
    Zero = 0
    String = 1
    Data = 2
    Pointer = 3
