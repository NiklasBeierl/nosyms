from typing import Callable, List, Dict
import dgl
import networkx as nx
import torch as t
from encoding import BlockType, VolatilitySymbolsEncoder
from encoding.block_types import blocks_to_tensor_truncate


def build_vol_symbols_graph(
    user_type_name: str,
    memory_encoder: VolatilitySymbolsEncoder,
    to_nx_graph: Callable[[List[BlockType], Dict[int, "type_descriptor"]], nx.Graph],
    blocks_to_tensor: Callable[[List[BlockType]], t.tensor] = blocks_to_tensor_truncate,
) -> dgl.DGLGraph:
    raise NotImplementedError
