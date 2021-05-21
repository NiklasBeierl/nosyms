from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple, Dict, List
from dgl import DGLHeteroGraph
from encoding.block_types import BlockType
from encoding.memory_blocks import MemoryEncoder, SandwichEncoder
from encoding.symbol_blocks import VolatilitySymbolsEncoder


class SymbolNodeId(NamedTuple):
    """
    Identifies nodes in a symbol graph. Since we can not store string information in dgl graphs directly, functions
    building graphs return a dict mapping these node ids onto indices in the graph.
    """

    type_descriptor: str
    chunk: int


class Pointer(NamedTuple):
    """
    Represents physical offset and target address of a pointer.
    """

    offset: int  # Where is the pointer?
    target: int  # Where does it point to?


class GraphBuilder(ABC):
    def __init__(self, *, pointer_size: int = 8):
        # Safeguard so we do not end throwing different architectures into the same graph. See _check_encoder.
        self.pointer_size = pointer_size

    def _check_encoder(self, encoder):
        if encoder.pointer_size != self.pointer_size:
            raise ValueError(f"Encoder {encoder} has incompatible pointer size.")

    @abstractmethod
    def create_type_graph(
        self, sym_encoder: VolatilitySymbolsEncoder, user_type_name: str
    ) -> Tuple[DGLHeteroGraph, Dict[SymbolNodeId, int]]:
        ...

    @abstractmethod
    def create_snapshot_graph(self, mem_encoder: MemoryEncoder, pointers: List[Pointer]) -> DGLHeteroGraph:
        ...


from encoding.ball import BallGraphBuilder

# TODO: Restructure
from encoding.memory_graph import build_memory_graph
from encoding.symbol_graph import build_vol_symbols_graph
