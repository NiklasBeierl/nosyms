from typing import NamedTuple
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


from encoding.ball import GraphBuilder, BallGraphBuilder
from encoding.memory_graph import build_memory_graph
from encoding.symbol_graph import build_vol_symbols_graph
