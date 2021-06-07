from collections import deque, defaultdict
from functools import partial, cached_property
import json
from multiprocessing import cpu_count
from typing import Tuple, List, Set, Deque, Iterable
from warnings import warn
from bidict import bidict, frozenbidict
import torch as t
import numpy as np
from dgl import DGLHeteroGraph, heterograph
from rbi_tree.tree import ITree
from pebble import ProcessPool
from encoding import (
    BlockType,
    SymbolNodeId,
    Pointer,
    GraphBuilder,
    MemoryEncoder,
    VolatilitySymbolsEncoder,
    BlockCompressor,
)
from encoding.symbol_blocks import UndefinedTypeError
from encoding.block_types import blocks_to_numpy
from hyperparams import BALL_RADIUS

# https://docs.dgl.ai/en/0.6.x/generated/dgl.heterograph.html?highlight=heterograph#dgl-heterograph
DglAdjacency = Tuple[Iterable[int], Iterable[int]]
# This representation of edges is used internally because its easier to work with at times.
EdgeList = List[Tuple[int, int]]

# This is used to to collect undefined type descriptors to make sure the corresponding error handling is not covering
# other errors. Therefore its only relevant if you are debugging.
_FAILED_SET = set()

# Encoding and compressing memory is batched for space / time efficiency
_ENCODE_BATCH_SIZE = 1000


def _edge_list_to_dgl(edges: EdgeList) -> DglAdjacency:
    if not edges:
        warn("Creating empty dgl adjacency, something might be wrong.")
        return [], []
    return tuple(zip(*edges))


def _grab_ball(blocks: np.array, offset: int, radius: int, pointer_size: int) -> np.array:
    size = (2 * radius) + pointer_size

    result = np.full(size, BlockType.Unknown, dtype=np.int8)
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


class BallEncoder(MemoryEncoder):
    """
    Encodes the immediate surroundings of given offset. (Usually the offset of a pointer)
    """

    def __init__(self, *args, pointers: List[int], radius: int = BALL_RADIUS, **kwargs):
        super().__init__(*args, **kwargs)
        self.pointers = pointers
        self.radius = radius

    @cached_property
    def as_numpy(self) -> np.array:
        raw = np.array(self.mmap, dtype=np.int8)
        result = np.full(len(raw), BlockType.Unknown, dtype=np.int8)
        printable = ((9 <= raw) & (raw <= 13)) | ((32 <= raw) & (raw <= 126))  # Printable ascii
        result[printable] = BlockType.String.value
        for pointer in self.pointers:
            result[pointer : pointer + self.pointer_size] = BlockType.Pointer.value
        return result

    def encode(self, offset: int) -> np.array:
        return _grab_ball(self.as_numpy, offset, self.radius, self.pointer_size)


def _compute_pointers_shard(intervals: List[Tuple[int, int, int]], pointers: List[Pointer]) -> EdgeList:
    chunks_tree = ITree()
    for interval in intervals:
        chunks_tree.insert(*interval)
    points_to_edges: EdgeList = []
    for source_id, p in pointers:
        for _, _, target_id in chunks_tree.find_at(p.target):
            points_to_edges.append((source_id, target_id))
    return points_to_edges


def compute_pointer_edges(intervals: List[Tuple[int, int, int]], sorted_pointers: List[Pointer]) -> DglAdjacency:
    # Scatter sorted_pointers across cores
    cpus = cpu_count() // 2  # TODO: rbi-tree causing OOM
    scattered_pointers = defaultdict(list)
    for i, pointer in enumerate(sorted_pointers):
        scattered_pointers[i % cpus].append((i, pointer))

    pool = ProcessPool(cpus)
    func = partial(_compute_pointers_shard, intervals)
    fut = pool.map(func, scattered_pointers.values())
    results: List[EdgeList] = list(fut.result())
    # Gather
    return _edge_list_to_dgl([item for sublist in results for item in sublist])


class BallGraphBuilder(GraphBuilder):
    def __init__(self, *args, radius: int = BALL_RADIUS, **kwargs):
        super(BallGraphBuilder, self).__init__(*args, **kwargs)
        self.radius = radius

    @cached_property
    def _max_centroid_dist(self):
        return self.pointer_size + (2 * self.radius)

    @staticmethod
    def _symobl_node_edges_to_dgl(
        edges: List[Tuple[SymbolNodeId, SymbolNodeId]], node_ids: bidict[SymbolNodeId, int]
    ) -> DglAdjacency:
        edges = [(node_ids[u], node_ids[v]) for u, v in edges]
        return _edge_list_to_dgl(edges)

    def _grab_ball(self, blocks: np.array, offset: int) -> np.array:
        return _grab_ball(blocks, offset, self.radius, self.pointer_size)

    def create_type_graph(
        self, sym_encoder: VolatilitySymbolsEncoder, user_type_name: str
    ) -> Tuple[DGLHeteroGraph, frozenbidict[SymbolNodeId, int]]:
        self._check_encoder(sym_encoder)

        # Results
        node_ids: bidict[SymbolNodeId, int] = bidict()
        in_struct_offsets = []
        block_data: List[np.array] = []
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
            except UndefinedTypeError:
                # Apparently volatility symbols sometimes have type_descriptors without corresponding types.
                _FAILED_SET.add(current_td)
                warn(f"Undefined type encountered while encoding symbols: {current_td}, see Readme for more details.")
                # Well, seems like we wont know.
                blocks, pointers = [BlockType.Unknown], {}

            # Doing this here to prevent adding current_td to the queue again if current_td contains a pointer to
            # another instance of the same type
            did_encode.add(current_td)

            blocks: np.array = blocks_to_numpy(blocks)

            if not pointers:
                block_data.append(self._grab_ball(blocks, 0))
                current_node = SymbolNodeId(current_td, 0)
                node_ids[current_node] = len(node_ids)
                in_struct_offsets.append(0)
                continue

            pointers = {offset: json.dumps(td) for offset, td in pointers.items()}
            sorted_pointers = list(sorted(pointers.items(), key=lambda p: p[0]))
            type_local_node_ids = {}
            for i, pointer in enumerate(sorted_pointers):
                offset, td = pointer
                block_data.append(self._grab_ball(blocks, offset))
                current_node = SymbolNodeId(current_td, i)
                type_local_node_ids[len(type_local_node_ids)] = current_node
                node_ids[current_node] = len(node_ids)
                in_struct_offsets.append(offset)
                if td not in did_encode and td not in to_encode:
                    to_encode.append(td)

                target_node = SymbolNodeId(td, 0)
                points_to_edges.append((current_node, target_node))

            for src, target in self._compute_precedes([offset for offset, _ in sorted_pointers]):
                precedes_edges.append((type_local_node_ids[src], type_local_node_ids[target]))

        points_to_edges: DglAdjacency = self._symobl_node_edges_to_dgl(points_to_edges, node_ids)
        precedes_edges: DglAdjacency = self._symobl_node_edges_to_dgl(precedes_edges, node_ids)
        follows_edges = precedes_edges[1], precedes_edges[0]  # You get the idea...
        graph: DGLHeteroGraph = heterograph(
            {
                ("chunk", "pointed_to_by", "chunk"): points_to_edges[::-1],
                ("chunk", "precedes", "chunk"): precedes_edges,
                ("chunk", "follows", "chunk"): follows_edges,
            }
        )
        graph.ndata["blocks"] = t.tensor(np.stack(block_data))
        graph.ndata["in_struct_offset"] = t.tensor(in_struct_offsets, dtype=t.int32)
        return graph, frozenbidict(node_ids)

    @staticmethod
    def _encode_and_compress(encoder: BallEncoder, compressor: BlockCompressor, pointers: List[int]) -> t.tensor:
        results = []
        for i in range((len(pointers) // _ENCODE_BATCH_SIZE) + 1):
            start, end = i * _ENCODE_BATCH_SIZE, min((i + 1) * _ENCODE_BATCH_SIZE, len(pointers))
            batch = []
            for p in pointers[start:end]:
                blocks = encoder.encode(p)
                batch.append(blocks)
            batch = t.tensor(np.stack(batch))
            if compressor:
                batch = compressor.compress_batch(batch)
            results.append(batch)
        return t.cat(results)

    def _compute_precedes(self, sorted_offsets: List[int]) -> EdgeList:
        """
        Compute precedes relationships between balls induced by offsets. (Alternative)
        Consider b1 to be preceding b2 if b1s centroid is before b2s centroid, their balls touch / overlap and there is
        no other centroid between b1 and b2.
        :param sorted_offsets: Sorted offsets of centroid balls
        :return: Precedes relationship as EdgeList
        """
        result: EdgeList = []
        i = 0
        while i < len(sorted_offsets) - 1:
            prev_offset = sorted_offsets[i]
            next_offset = sorted_offsets[i + 1]
            if next_offset <= prev_offset + self._max_centroid_dist:
                result.append((i, i + 1))
            i += 1
        return result

    # Surprisingly, results tend to be worse when using this logic.
    def _compute_precedes_alt(self, sorted_offsets: List[int]) -> EdgeList:
        """
        Compute precedes relationships between balls induced by offsets. (Alternative)
        Consider b1 to be preceding b2 if b1s centroid is before b2s centroid and their balls touch / overlap.
        :param sorted_offsets: Sorted offsets of centroid balls
        :return: Precedes relationship as EdgeList
        """
        result: EdgeList = []
        for p_id, prev_offset in enumerate(sorted_offsets[:-1]):
            n_id = p_id + 1
            max_offset = prev_offset + self._max_centroid_dist
            while n_id < len(sorted_offsets) and sorted_offsets[n_id] <= max_offset:
                result.append((p_id, n_id))
                n_id += 1

        return result

    def create_snapshot_graph(
        self, mem_encoder: BallEncoder, pointers: List[Pointer], compressor: BlockCompressor = None
    ) -> DGLHeteroGraph:
        # TODO: Check if Mem encoder has same radius as graph builder
        self._check_encoder(mem_encoder)

        # Sort pointers by their offset, ascending
        sorted_pointers = list(sorted(pointers, key=lambda p: p.offset))

        # The min and max could be replaced by something more efficient since the pointers are sorted.
        starts = np.array([max(0, p.offset - self.radius) for p in sorted_pointers], dtype=np.int64)
        end_offset = self.pointer_size + self.radius
        ends = np.array([min(mem_encoder.size, p.offset + end_offset) for p in sorted_pointers], dtype=np.int64)

        intervals = list(zip(starts, ends, range(len(pointers))))

        points_to_edges = compute_pointer_edges(intervals, sorted_pointers)
        precedes_edges = _edge_list_to_dgl(self._compute_precedes([p.offset for p in sorted_pointers]))
        follows_edges = precedes_edges[1], precedes_edges[0]
        graph = heterograph(
            {
                ("chunk", "pointed_to_by", "chunk"): points_to_edges[::-1],
                ("chunk", "precedes", "chunk"): precedes_edges,
                ("chunk", "follows", "chunk"): follows_edges,
            }
        )
        graph.ndata["blocks"] = self._encode_and_compress(mem_encoder, compressor, [p.offset for p in sorted_pointers])
        graph.ndata["pointer_offset"] = t.tensor([p.offset for p in sorted_pointers], dtype=t.int64)
        graph.ndata["start"] = t.tensor(starts)
        graph.ndata["end"] = t.tensor(ends)
        return graph
