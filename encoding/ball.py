from collections import deque
import json
from typing import Tuple, Dict, List, Set, Deque
from warnings import warn
from bidict import bidict, frozenbidict
import torch as t
from dgl import DGLHeteroGraph, heterograph
from encoding import BlockType, SymbolNodeId, GraphBuilder, MemoryEncoder, VolatilitySymbolsEncoder

BALL_RADIUS = 100

_BLOCKS_TO_INT = {
    bt: bt.value for bt in BlockType,
}
_BLOCKS_TO_INT[None] = 0


def _blocks_to_tensor(blocks: List[BlockType]) -> t.tensor:
    result = t.zeros(len(blocks), dtype=t.int8)
    for i, block in enumerate(blocks):
        result[i] = _BLOCKS_TO_INT[block]
    return result


def _grab_chunk(blocks: t.tensor, offset: int, radius: int, pointer_size: int) -> t.tensor:
    size = (2 * radius) + pointer_size
    result = t.zeros(size, dtype=t.int8)
    start = offset - radius
    end = offset + pointer_size + radius
    chunk_offset = 0
    if offset < radius:
        start = 0
        chunk_offset = radius - offset
    if end > len(blocks):
        end = len(blocks)
    chunk = blocks[start:end]
    result[chunk_offset : chunk_offset + len(chunk)] = chunk
    return result


def _prepare_edges_for_dgl(
    edges: List[Tuple[SymbolNodeId, SymbolNodeId]], node_ids: Dict[SymbolNodeId, int], did_encode: Set[str]
):
    edges = [(u, v) for u, v in edges if u.type_descriptor in did_encode and v.type_descriptor in did_encode]
    edges = [(node_ids[u], node_ids[v]) for u, v in edges]
    return tuple(zip(*edges))


class BallGraphBuilder(GraphBuilder):
    def __init__(self, *args, radius: int = BALL_RADIUS, **kwargs):
        super(BallGraphBuilder, self).__init__(*args, **kwargs)
        self.radius = radius

    def _grab_chunk(self, blocks, offset):
        return _grab_chunk(blocks, offset, self.radius, self.pointer_size)

    def create_type_graph(
        self, sym_encoder: VolatilitySymbolsEncoder, user_type_name: str
    ) -> Tuple[DGLHeteroGraph, frozenbidict]:
        self._check_encoder(sym_encoder)

        # Results
        node_ids: Dict[SymbolNodeId, int] = bidict()
        block_data: List[t.tensor] = []
        points_to_edges: List[Tuple[SymbolNodeId, SymbolNodeId]] = []
        precedes_edges: List[Tuple[SymbolNodeId, SymbolNodeId]] = []

        # Stuff for the while loop
        type_descriptor = json.dumps({"kind": "struct", "name": user_type_name})
        to_encode: Deque[str] = deque([type_descriptor])
        did_encode: Set[str] = set()
        while to_encode:
            current_td = to_encode.pop()
            try:
                blocks, pointers = sym_encoder.encode_type_descriptor(json.loads(current_td))
            except KeyError:
                # Apparently volatility symbols sometimes have type_descriptors without corresponding types.
                warn(f"Undefined type encountered while encoding symbols: {current_td}, see Readme for more details.")
                # We will drop the corresponding edges from edges later
                continue
            # Doing this here to prevent adding current_td to the queue again if current_td contains a pointer to
            # another instance of the same type
            did_encode.add(current_td)

            blocks = _blocks_to_tensor(blocks)
            pointers = {offset: json.dumps(td) for offset, td in pointers.items()}

            if not pointers:
                block_data.append(self._grab_chunk(blocks, 0))
                current_node = SymbolNodeId(current_td, 0)
                node_ids[current_node] = len(node_ids)
                continue

            last_node = None
            for i, pointer in enumerate(pointers.items()):
                offset, td = pointer
                block_data.append(self._grab_chunk(blocks, offset))
                current_node = SymbolNodeId(current_td, i)
                node_ids[current_node] = len(node_ids)
                if td not in did_encode and td not in to_encode:
                    to_encode.append(td)

                target_node = SymbolNodeId(td, 0)
                points_to_edges.append((current_node, target_node))
                if last_node:
                    precedes_edges.append((last_node, current_node))
                last_node = current_node

        points_to_edges = _prepare_edges_for_dgl(points_to_edges, node_ids, did_encode)
        precedes_edges = _prepare_edges_for_dgl(precedes_edges, node_ids, did_encode)
        follows_edges = precedes_edges[1], precedes_edges[0]  # You get the idea...
        graph = heterograph(
            {
                ("chunk", "points_to", "chunk"): points_to_edges,
                ("chunk", "precedes", "chunk"): precedes_edges,
                ("chunk", "follows", "chunk"): follows_edges,
            }
        )
        graph.ndata["blocks"] = t.stack(block_data)
        return graph, frozenbidict(node_ids)

    def create_snapshot_graph(self, mem_encoder: MemoryEncoder, pointers) -> DGLHeteroGraph:
        self._check_encoder(mem_encoder)
        raise NotImplementedError("TODO")
