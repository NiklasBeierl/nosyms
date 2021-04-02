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


class BlockType:
    @_singleton
    class Zero:
        pass

    @_singleton
    class String:
        pass

    @_singleton
    class Data:
        pass

    class Pointer:
        def __init__(self, target: int):
            self.target: int = target

        def __repr__(self):
            return f"Pointer to {hex(self.target)}"
