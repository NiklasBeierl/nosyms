from collections import deque, defaultdict
from functools import partial, cached_property
import json
from multiprocessing import cpu_count
from typing import Tuple, List, Set, Deque, Iterable, Dict, NamedTuple
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

# This representation of edges is used internally because its easier to work with than DglAdjacency at times
EdgeList = List[Tuple[int, int]]


# Pointer holding node_id of a corresponding chunk instead of his offset
class ChunkPointer(NamedTuple):
    node_id: int
    target: int


# Slice / chunk of memory, ranging from start to end and having a node_id in the dgl graph
class Chunk(NamedTuple):
    start: int
    end: int
    node_id: int


# This is used to to collect undefined type descriptors to make sure the corresponding error handling is not covering
# other errors. Therefore its only relevant if you are debugging.
_FAILED_SET = set()

# Encoding and compressing memory is batched for space / time efficiency
_ENCODE_BATCH_SIZE = 1000


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


def _edge_list_to_dgl(edges: EdgeList) -> DglAdjacency:
    if not edges:
        warn("Creating empty dgl adjacency, something might be wrong.")
        return [], []
    return tuple(zip(*edges))


def _compute_pointers_shard(chunks: List[Chunk], pointers: List[ChunkPointer]) -> EdgeList:
    """
    Handle computation of compute_pointer_edges on a subset of the offsets. Refer to calling methods docs.
    :param chunks: Memory chunks as triples: start, end , node_id
    :param pointers: List of node_id, pointer tuples
    :return: EdgeList identified edges
    """
    chunks_tree = ITree()
    for chunk in chunks:
        chunks_tree.insert(*chunk)
    points_to_edges: EdgeList = []
    for node_id, target_address in pointers:
        for _, _, target_id in chunks_tree.find_at(target_address):
            points_to_edges.append((node_id, target_id))
    return points_to_edges


def _compute_pointer_edges(chunks: List[Chunk], chunk_pointers: List[ChunkPointer]) -> DglAdjacency:
    """
    Compute dgl adjacency from chunks and ChunkPointers.
    :param chunks: Memory chunks.
    :param chunk_pointers: List of Chunk Pointers.
    :return: Dgl adjacency describing the points_to relationship
    """

    # Scatter chunk_pointers across cores
    cpus = cpu_count() // 2  # TODO: rbi-tree causing OOM
    scattered_pointers = defaultdict(list)
    for i, pointer in enumerate(chunk_pointers):
        scattered_pointers[i % cpus].append(pointer)

    pool = ProcessPool(cpus)
    func = partial(_compute_pointers_shard, chunks)
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
        """
        Maximum distance two points can have in order for their balls to touch / overlap.
        """
        return self.pointer_size + (2 * self.radius)

    def _grab_ball(self, blocks: np.array, offset: int) -> np.array:
        """
        Alias of _grab_ball with pointer size and radius applied.
        """
        return _grab_ball(blocks, offset, self.radius, self.pointer_size)

    def _ball_from_offset(self, offset: int) -> Tuple[int, int]:
        """
        Create ball around offset according to radius and pointer size.
        """
        return offset - self.pointer_size, offset + self.radius + self.pointer_size

    # This might seem overkill, but it makes implementing congruent encodings a lot simpler.
    def _simulate_addresses(
        self, types_chunks: List[Dict[int, int]], node_id_points_to: Dict[int, int]
    ) -> Tuple[List[int], List[Chunk], List[ChunkPointer]]:
        """
        Provides simulated chunk addresses so the same functions for adjacency calculation can be used when encoding
        memory and symbols. The addresses are created in such a way that every type has one "instance". The chunks
        within a type will overlap just as they would in a real snapshot.
        The chunks of two different types will never overlap.
        :param types_chunks: List of Dict["offset", "node_id]. Every Dict describes the offsets of chunks within a type.
        :param node_id_points_to: Incomplete points_to relationship: These edges only contain the edges from a chunk to
        the first chunk of the type they belong to.
        :return: Simulated centroids, chunks and chunk pointers
        """
        node_addresses: Dict[int, int] = {}  # Dict["node id", "address in simulated address space"]
        chunks: List[Chunk] = []
        type_base_address = self.radius
        for type_chunks in types_chunks:
            for offset, node_id in type_chunks.items():
                node_addresses[node_id] = type_base_address
                node_offset = type_base_address + offset
                chunk = Chunk(*self._ball_from_offset(node_offset), node_id)
                chunks.append(chunk)
            # upper bound of last chunk plus safe distance
            type_base_address = chunk.end + self._max_centroid_dist + 1

        chunk_pointers = [
            ChunkPointer(node_id, node_addresses[target]) for node_id, target in node_id_points_to.items()
        ]
        sorted_centroids = list(sorted(node_addresses.values()))
        return sorted_centroids, chunks, chunk_pointers

    def create_type_graph(
        self, sym_encoder: VolatilitySymbolsEncoder, user_type_name: str
    ) -> Tuple[DGLHeteroGraph, frozenbidict[SymbolNodeId, int]]:
        self._check_encoder(sym_encoder)

        # Results
        node_ids: bidict[SymbolNodeId, int] = bidict()
        points_to: Dict[SymbolNodeId, SymbolNodeId] = {}
        # Dict["node_id of first chunk", Dict["offset in struct", "node_id"]
        types_chunks: List[Dict[int, int]] = []
        in_struct_offsets = []
        block_vectors: List[np.array] = []

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
                blocks, pointers = [BlockType.Unknown], {}  # Well, seems like we wont know.

            # Doing this here prevents adding current_td to the queue again if current_td contains a pointer to
            # another instance of the same type
            did_encode.add(current_td)

            blocks: np.array = blocks_to_numpy(blocks)
            pointers = {offset: json.dumps(td) for offset, td in pointers.items()}
            if 0 not in pointers:  # Make sure the start of the struct gets a chunk
                pointers[0] = None
            sorted_pointers = list(sorted(pointers.items(), key=lambda p: p[0]))
            type_chunks: Dict[int, int] = {}  # Dict["offset", "node_id"]
            for i, pointer in enumerate(sorted_pointers):
                offset, target_td = pointer
                block_vectors.append(self._grab_ball(blocks, offset))
                in_struct_offsets.append(offset)
                current_node = SymbolNodeId(current_td, i)
                node_id = len(node_ids)
                node_ids[current_node] = type_chunks[offset] = node_id
                if target_td:
                    target_node = SymbolNodeId(target_td, 0)
                    points_to[current_node] = target_node
                    if target_td not in did_encode and target_td not in to_encode:
                        to_encode.append(target_td)
            types_chunks.append(type_chunks)

        node_id_points_to = {node_ids[u]: node_ids[v] for u, v in points_to.items()}
        sorted_centroids, chunks, pointers = self._simulate_addresses(types_chunks, node_id_points_to)

        points_to_edges = _compute_pointer_edges(chunks, pointers)
        precedes_edges = self._compute_precedes(sorted_centroids)
        follows_edges = precedes_edges[1], precedes_edges[0]  # Inverse of precedes
        graph: DGLHeteroGraph = heterograph(
            {
                ("chunk", "pointed_to_by", "chunk"): points_to_edges[::-1],
                ("chunk", "precedes", "chunk"): precedes_edges,
                ("chunk", "follows", "chunk"): follows_edges,
            }
        )
        graph.ndata["blocks"] = t.tensor(np.stack(block_vectors))
        graph.ndata["in_struct_offset"] = t.tensor(in_struct_offsets, dtype=t.int32)
        return graph, frozenbidict(node_ids)

    @staticmethod
    def _encode_and_compress(encoder: BallEncoder, compressor: BlockCompressor, offsets: List[int]) -> t.tensor:
        results = []
        for i in range((len(offsets) // _ENCODE_BATCH_SIZE) + 1):
            start, end = i * _ENCODE_BATCH_SIZE, min((i + 1) * _ENCODE_BATCH_SIZE, len(offsets))
            batch = []
            for p in offsets[start:end]:
                blocks = encoder.encode(p)
                batch.append(blocks)
            batch = t.tensor(np.stack(batch))
            if compressor:
                batch = compressor.compress_batch(batch)
            results.append(batch)
        return t.cat(results)

    def _compute_precedes(self, sorted_centroids: List[int]) -> DglAdjacency:
        """
        Compute precedes relationships between balls induced by offsets. (Alternative)
        Consider b1 to be preceding b2 if b1s centroid is before b2s centroid, their balls touch / overlap and there is
        no other centroid between b1 and b2.
        :param sorted_centroids: Sorted offsets of centroid balls
        :return: Precedes relationship as EdgeList
        """
        result: EdgeList = []
        i = 0
        while i < len(sorted_centroids) - 1:
            prev_offset = sorted_centroids[i]
            next_offset = sorted_centroids[i + 1]
            if next_offset <= prev_offset + self._max_centroid_dist:
                result.append((i, i + 1))
            i += 1
        return _edge_list_to_dgl(result)

    # Surprisingly, results tend to be worse when using this logic.
    def _compute_precedes_alt(self, sorted_centroids: List[int]) -> DglAdjacency:
        """
        Compute precedes relationships between balls induced by offsets. (Alternative)
        Consider b1 to be preceding b2 if b1s centroid is before b2s centroid and their balls touch / overlap.
        :param sorted_centroids: Sorted offsets of centroid balls
        :return: Precedes relationship as EdgeList
        """
        result: EdgeList = []
        for p_id, prev_offset in enumerate(sorted_centroids[:-1]):
            n_id = p_id + 1
            max_offset = prev_offset + self._max_centroid_dist
            while n_id < len(sorted_centroids) and sorted_centroids[n_id] <= max_offset:
                result.append((p_id, n_id))
                n_id += 1

        return _edge_list_to_dgl(result)

    def create_snapshot_graph(
        self, mem_encoder: BallEncoder, pointers: List[Pointer], compressor: BlockCompressor = None
    ) -> DGLHeteroGraph:
        self._check_encoder(mem_encoder)
        if mem_encoder.radius != self.radius:
            raise ValueError("Passed memory encoder needs to have the same radius as the graph builder.")

        # All offsets of interest, they either point to something or are pointed to
        sorted_centroids: List[int] = list(sorted(set([p.offset for p in pointers] + [p.target for p in pointers])))

        # Chunks (memory segments)
        starts = np.array([max(0, c - self.radius) for c in sorted_centroids], dtype=np.int64)
        end_offset = self.pointer_size + self.radius
        ends = np.array([min(mem_encoder.size, c + end_offset) for c in sorted_centroids], dtype=np.int64)
        chunks: List[Chunk] = [Chunk(s, e, i) for s, e, i in zip(starts, ends, range(len(sorted_centroids)))]

        # Pointers always belong to a chunk, their offsets are discarded in favor of the node_id after this point
        p_dict = {p.offset: p.target for p in pointers}
        chunk_pointers: List[ChunkPointer] = [
            ChunkPointer(i, p_dict[offset]) for i, offset in enumerate(sorted_centroids) if offset in p_dict
        ]

        # TODO: This call leaks some 30 gigs of memory, wtf?
        points_to_edges = _compute_pointer_edges(chunks, chunk_pointers)
        precedes_edges = self._compute_precedes(sorted_centroids)
        follows_edges = precedes_edges[1], precedes_edges[0]  # Inverse of precedes

        graph = heterograph(
            {
                ("chunk", "pointed_to_by", "chunk"): points_to_edges[::-1],
                ("chunk", "precedes", "chunk"): precedes_edges,
                ("chunk", "follows", "chunk"): follows_edges,
            }
        )

        graph.ndata["blocks"] = self._encode_and_compress(mem_encoder, compressor, sorted_centroids)
        # In most cases, start and end could be derived from offset and radius
        # But not at the boundaries of the snapshot and in the future maybe also not around page boundaries.
        graph.ndata["offset"] = t.tensor(sorted_centroids, dtype=t.int64)
        graph.ndata["start"] = t.tensor(starts, dtype=t.int64)
        graph.ndata["end"] = t.tensor(ends, dtype=t.int64)
        return graph
