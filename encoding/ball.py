from collections import deque, defaultdict
from collections.abc import Sequence
from functools import partial
import json
from typing import Tuple, Dict, List, Set, Deque
from warnings import warn
from multiprocessing import cpu_count
from bidict import bidict, frozenbidict
import torch as t
import numpy as np
from dgl import DGLHeteroGraph, heterograph
from rbi_tree.tree import ITree
from pebble import ProcessPool
from encoding import BlockType, SymbolNodeId, Pointer, GraphBuilder, MemoryEncoder, VolatilitySymbolsEncoder
from encoding.block_types import blocks_to_tensor
from hyperparams import BALL_RADIUS


def _grab_ball(blocks: 'Sequence["uint8"]', offset: int, radius: int, pointer_size: int) -> t.tensor:
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

    if t.is_tensor(blocks):
        blocks = blocks.numpy()
    chunk = blocks[start:end]
    result[chunk_offset : chunk_offset + len(chunk)] = t.tensor(chunk)
    return result


class BallEncoder(MemoryEncoder):
    """
    Encodes the immediate surroundings of given offset. (Usually the offset of a pointer)
    """

    def __init__(self, *args, radius: int = BALL_RADIUS, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius

    def _grab_ball(self, blocks, offset):
        return _grab_ball(blocks, offset, self.radius, self.pointer_size)

    def encode(self, offset: int) -> t.tensor:
        chunk = self._grab_ball(self.as_numpy, offset)
        chunk[offset : offset + self.pointer_size] = BlockType.Pointer.value
        return chunk


def _compute_pointers_shard(intervals: List[Tuple[int, int, int]], pointers: List[Pointer]):
    chunks_tree = ITree()
    for interval in intervals:
        chunks_tree.insert(*interval)
    points_to_edges = []
    for source_id, p in pointers:
        for _, _, target_id in chunks_tree.find_at(p.target):
            points_to_edges.append((source_id, target_id))
    return points_to_edges


def _compute_pointers(intervals: List[Tuple[int, int, int]], sorted_pointers: List[Pointer]):
    # Scatter pointers accross cores
    cpus = cpu_count()
    scattered_pointers = defaultdict(list)
    for i, pointer in enumerate(sorted_pointers):
        scattered_pointers[i % cpus].append((i, pointer))

    pool = ProcessPool(cpus)
    func = partial(_compute_pointers_shard, intervals)
    result = pool.map(func, scattered_pointers.values())
    result = list(result.result())
    # Gather
    result = [item for sublist in result for item in sublist]
    return result


class BallGraphBuilder(GraphBuilder):
    def __init__(self, *args, radius: int = BALL_RADIUS, **kwargs):
        super(BallGraphBuilder, self).__init__(*args, **kwargs)
        self.radius = radius

    @staticmethod
    def _prepare_edges_for_dgl(
        edges: List[Tuple[SymbolNodeId, SymbolNodeId]], node_ids: Dict[SymbolNodeId, int], did_encode: Set[str]
    ):
        edges = [(u, v) for u, v in edges if u.type_descriptor in did_encode and v.type_descriptor in did_encode]
        edges = [(node_ids[u], node_ids[v]) for u, v in edges]
        return tuple(zip(*edges))

    def _grab_ball(self, blocks, offset):
        return _grab_ball(blocks, offset, self.radius, self.pointer_size)

    def create_type_graph(
        self, sym_encoder: VolatilitySymbolsEncoder, user_type_name: str
    ) -> Tuple[DGLHeteroGraph, Dict[SymbolNodeId, int]]:
        self._check_encoder(sym_encoder)

        # Results
        node_ids: Dict[SymbolNodeId, int] = bidict()
        in_struct_offsets = []
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

            blocks = blocks_to_tensor(blocks)
            pointers = {offset: json.dumps(td) for offset, td in pointers.items()}

            if not pointers:
                block_data.append(self._grab_ball(blocks, 0))
                current_node = SymbolNodeId(current_td, 0)
                node_ids[current_node] = len(node_ids)
                in_struct_offsets.append(0)
                continue

            last_node = None
            for i, pointer in enumerate(pointers.items()):
                offset, td = pointer
                block_data.append(self._grab_ball(blocks, offset))
                current_node = SymbolNodeId(current_td, i)
                node_ids[current_node] = len(node_ids)
                in_struct_offsets.append(offset)
                if td not in did_encode and td not in to_encode:
                    to_encode.append(td)

                target_node = SymbolNodeId(td, 0)
                points_to_edges.append((current_node, target_node))
                if last_node:
                    precedes_edges.append((last_node, current_node))
                last_node = current_node

        points_to_edges = self._prepare_edges_for_dgl(points_to_edges, node_ids, did_encode)
        precedes_edges = self._prepare_edges_for_dgl(precedes_edges, node_ids, did_encode)
        follows_edges = precedes_edges[1], precedes_edges[0]  # You get the idea...
        graph = heterograph(
            {
                ("chunk", "pointed_to_by", "chunk"): points_to_edges[::-1],
                ("chunk", "precedes", "chunk"): precedes_edges,
                ("chunk", "follows", "chunk"): follows_edges,
            }
        )
        graph.ndata["blocks"] = t.stack(block_data)
        graph.ndata["in_struct_offset"] = t.tensor(in_struct_offsets, dtype=t.int32)
        return graph, frozenbidict(node_ids)

    def create_snapshot_graph(self, mem_encoder: BallEncoder, pointers: List[Pointer]) -> DGLHeteroGraph:
        self._check_encoder(mem_encoder)

        # Sort pointers by their offset, ascending
        sorted_pointers = list(sorted(pointers, key=lambda p: p.offset))

        # The min and max could be replaced by something more efficient since the pointers are sorted.
        starts = np.array([max(0, p.offset - self.radius) for p in sorted_pointers], dtype=np.int64)
        end_offset = self.pointer_size + self.radius
        ends = np.array([min(mem_encoder.size, p.offset + end_offset) for p in sorted_pointers], dtype=np.int64)

        intervals = list(zip(starts, ends, range(len(pointers))))

        points_to_edges = _compute_pointers(intervals, sorted_pointers)
        points_to_edges = tuple(zip(*points_to_edges))
        precedes_edges = list(range(len(pointers) - 1)), list(range(1, len(pointers)))
        follows_edges = precedes_edges[1], precedes_edges[0]
        graph = heterograph(
            {
                ("chunk", "pointed_to_by", "chunk"): points_to_edges[::-1],
                ("chunk", "precedes", "chunk"): precedes_edges,
                ("chunk", "follows", "chunk"): follows_edges,
            }
        )
        graph.ndata["blocks"] = t.stack([mem_encoder.encode(p.offset) for p in sorted_pointers])
        graph.ndata["pointer_offset"] = t.tensor([p.offset for p in sorted_pointers], dtype=t.int64)
        graph.ndata["start"] = t.tensor(starts)
        graph.ndata["end"] = t.tensor(ends)
        return graph
