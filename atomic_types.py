# TODO: More elegant way to make "static types" Singletons?

class AtomicType:
    class Padding:
        pass

    Padding = Padding()

    class String:
        pass

    String = String()

    class Data:
        pass

    Data = Data()

    class Pointer:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return f"Pointer to: {self.name}"
