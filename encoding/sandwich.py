from collections import deque
import json
from enum import Enum
from string import printable
from typing import Tuple, List, Iterable, Callable, NamedTuple, Dict
from warnings import warn
import dgl
import torch as t
import networkx as nx
from encoding import VolatilitySymbolsEncoder, SymbolNodeId, BlockType, MemoryEncoder, Pointer

NODE_MAX_LEN = 53  # Mean length + 1 std of task_struct related nodes in my dataset.


def blocks_to_tensor_truncate(blocks: List[BlockType], tensor_length: int = NODE_MAX_LEN) -> t.tensor:
    """
    Convert list of BlockTypes to tensor of shape (tensor_length, len(BlockType)). Every row in the tensor "one-hot"
    encodes the corresponding block in `blocks`. If `blocks` is longer than `tensor_length`, it is truncated.
    If it is shorter, the the remaining "rows" will be all 0.
    :param blocks: List of blocks to be encoded.
    :param tensor_length: Length of the resulting tensor.
    :return: tensor of shape (tensor_length, len(BlockType))
    """
    num_ts = len(BlockType)
    output = t.zeros(tensor_length * len(BlockType))
    for i, b in enumerate(blocks[:tensor_length]):
        output[i * num_ts + b] = 1
    return output.to_sparse()


def _determine_byte_type(byte: int) -> BlockType:
    if chr(byte) in printable:
        return BlockType.String
    else:
        return BlockType.Data


class SandwichEncoder(MemoryEncoder):
    """
    Encodes memory between "start and end". Assumes that "start" points to the first and "end"
    to the last byte of a pointer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bread = [BlockType.Pointer] * self.pointer_size

    def encode(self, start, end):
        # TODO: What about start == 0?
        if start + self.pointer_size == end:
            return [BlockType.Pointer, BlockType.Pointer] * self.pointer_size
        else:
            encoded = [
                _determine_byte_type(char)
                for char in self.mmap[start + self.pointer_size : end - self.pointer_size + 1]
            ]
            return self._bread + encoded + self._bread


class MemSlice(NamedTuple):
    """
    Represent slices of memory in :func:`data_and_surrounding_pointers`
    """

    id: int
    start: int
    end: int


class MemRelations(Enum):
    """
    Represent relationships between memory slices in :func:`data_and_surrounding_pointers`
    """

    FOLLOWS = "FOLLOWS"
    PRECEDS = "PRECEDS"
    BELOW_POINTED_TO = "BELOW_POINTED_TO"
    ABOVE_POINTED_TO = "ABOVE_POINTED_TO"

    def __str__(self):
        return self.value


def data_and_surrounding_pointers(pointers: List[Pointer], pointer_size: int) -> nx.Graph:
    """
    TODO
    :param pointers:
    :return:
    """
    nodes = []
    p_offset_to_node_id = {}
    graph = nx.DiGraph()
    start_offset = 0
    for i, p in enumerate(sorted(pointers, key=lambda pointer: pointer.offset)):
        real_end = p.offset + pointer_size - 1  # Inclusive end, including the "end" pointer
        nodes.append(MemSlice(i, start_offset, real_end))
        p_offset_to_node_id[p.offset] = i
        graph.add_node(i, start=start_offset, end=real_end)
        if nodes:
            graph.add_edge(i, nodes[i - 1].id, type=MemRelations.FOLLOWS)
            graph.add_edge(nodes[i - 1].id, i, type=MemRelations.PRECEDS)
        start_offset = p.offset

    node_index = 0
    for offset, target in sorted(pointers, key=lambda pointer: pointer.target):
        while node_index < len(nodes) and nodes[node_index].end < target:
            node_index += 1
        if node_index == len(nodes):
            break
        pointing_node = p_offset_to_node_id[offset]

        graph.add_edge(
            nodes[node_index].id,
            pointing_node,
            target=target,
            type=MemRelations.BELOW_POINTED_TO,
        )
        if pointing_node + 1 < len(nodes):
            graph.add_edge(
                nodes[node_index].id,
                pointing_node + 1,
                target=target,
                type=MemRelations.ABOVE_POINTED_TO,
            )

    return graph


def _extract_adjacency(graph: nx.DiGraph, relation_type) -> Tuple[Iterable, Iterable]:
    """
    Extract adjacency information from an nx.DiGraph in DGLS "tuple of node-tensors" format for one "type" of relation.
    Only Edges with attribute `type == relation_type` are considered.
    See: https://docs.dgl.ai/generated/dgl.heterograph.html.
    :param graph: Graph to extract adjacency from.
    :param relation_type: `type` to filter edges by.
    :return: Adjacency in "tuple of node-tensors" format.
    """
    edge_attrs = nx.get_edge_attributes(graph, "type")
    relevant_edges = [edge for edge in graph.edges if edge_attrs[edge] == relation_type]
    return tuple(zip(*relevant_edges))


def build_memory_graph(
    pointers: List[Pointer],
    memory_encoder: MemoryEncoder,
    to_nx_graph: Callable[[List[Pointer]], nx.Graph] = data_and_surrounding_pointers,
    blocks_to_tensor: Callable[[List[BlockType]], t.tensor] = blocks_to_tensor_truncate,
) -> dgl.DGLGraph:
    """
    TODO
    :param pointers:
    :param memory_encoder:
    :param to_nx_graph:
    :param blocks_to_tensor:
    :return:
    """
    nx_g = to_nx_graph(pointers, memory_encoder.pointer_size)
    relation_types = set(nx.get_edge_attributes(nx_g, "type").values())
    dgl_graph_data = {("slice", str(t), "slice"): _extract_adjacency(nx_g, t) for t in relation_types}
    dgl_graph = dgl.heterograph(dgl_graph_data)
    blocks = {id: memory_encoder.encode(**data) for id, data in nx_g.nodes.data()}
    blocks = {id: blocks_to_tensor(blocks_list) for id, blocks_list in blocks.items()}
    blocks = t.stack([blocks[id] for id in range(len(nx_g))])
    start = t.tensor([nx_g.nodes[id]["start"] for id in range(len(nx_g))])
    end = t.tensor([nx_g.nodes[id]["end"] for id in range(len(nx_g))])
    dgl_graph.ndata["blocks"] = blocks
    dgl_graph.ndata["start"] = start
    dgl_graph.ndata["end"] = end
    return dgl_graph


def _pointer_delimited_chunks(blocks: List[BlockType], pointer_size: int) -> List[Tuple[int, int]]:
    """
    Generate list of start, end interval from list of blocks, where every pointer block marks end of the previous and
    start of the next interval.
    :param blocks: List of blocks to generate intervals from.
    :return: List of start, end tuples
    """
    result = []
    last = 0
    current = 0
    while current < len(blocks):
        block = blocks[current]
        if block == BlockType.Pointer:
            result.append((last, current + pointer_size - 1))
            last = current
            current += pointer_size
        else:
            current += 1
    if last != (len(blocks) - 1):
        result.append((last, len(blocks) - 1))
    return result


def data_and_surrounding_pointers(
    type_descriptor: "type_descriptor", memory_encoder: VolatilitySymbolsEncoder
) -> nx.DiGraph:
    graph = nx.DiGraph()
    to_encode = deque([json.dumps(type_descriptor)])
    did_encode = set()
    deferred_edges = []
    while to_encode:
        curr_td_str = to_encode.pop()
        try:
            blocks, pointers = memory_encoder.encode_type_descriptor(json.loads(curr_td_str))
        except KeyError:
            # Apparently volatility symbols sometimes have type_descriptors without corresponding types.
            warn(f"Undefined type encountered while encoding symbols: {curr_td_str}, see Readme for more details.")
            # We will drop the corresponding edges from deferred_edges later
            continue
        pointers = {offset: json.dumps(td) for offset, td in pointers.items()}
        if 0 in pointers and pointers[0] not in did_encode and pointers[0] not in to_encode:
            to_encode.append(pointers[0])

        last_node = None
        for start, end in _pointer_delimited_chunks(blocks, memory_encoder.pointer_size):
            current_node = SymbolNodeId(curr_td_str, start)
            graph.add_node(current_node, start=start, end=end)
            if last_node:
                graph.add_edge(last_node, current_node, type=MemRelations.PRECEDS)
                graph.add_edge(current_node, last_node, type=MemRelations.FOLLOWS)

            end_pointer = end - memory_encoder.pointer_size + 1
            if end_pointer in pointers:
                # This will also cover all the of the next node
                if pointers[end_pointer] not in did_encode and pointers[end_pointer] not in to_encode:
                    to_encode.append(pointers[end_pointer])
                deferred_edges.append(
                    (SymbolNodeId(pointers[end_pointer], 0), current_node, MemRelations.BELOW_POINTED_TO)
                )
            if start in pointers:
                deferred_edges.append((SymbolNodeId(pointers[start], 0), current_node, MemRelations.ABOVE_POINTED_TO))
            last_node = current_node
        did_encode.add(curr_td_str)
    for u, v, rel_type in deferred_edges:
        if u in graph:  # We might have encountered a key error above, in that case u can not be used.
            graph.add_edge(u, v, type=rel_type)
    return graph


def _map_out_nodes(
    dgl_graph_data: Dict, node_labels: Dict[SymbolNodeId, int]
) -> Tuple[Dict, Dict[SymbolNodeId, int], Dict[SymbolNodeId, int]]:
    new_dgl_data = {}
    local_node_ids: Dict[SymbolNodeId, int] = {}
    new_node_labels = node_labels.copy()
    next_int_label = max(new_node_labels.values(), default=-1) + 1
    for relation, adjacency in dgl_graph_data.items():
        new_adjacency = []
        for u, v in zip(*adjacency):
            for n in [u, v]:
                if n not in new_node_labels:
                    new_node_labels[n] = next_int_label
                    next_int_label += 1
                if n not in local_node_ids:
                    local_node_ids[n] = len(local_node_ids)
            new_adjacency.append((local_node_ids[u], local_node_ids[v]))
        new_dgl_data[relation] = tuple(zip(*new_adjacency))

    return new_dgl_data, new_node_labels, local_node_ids


def build_vol_symbols_graph(
    user_type_name: str,
    symbol_encoder: VolatilitySymbolsEncoder,
    node_labels: Dict[Tuple[str, int], int] = None,
    to_nx_graph: Callable[["type_descriptor", VolatilitySymbolsEncoder], nx.Graph] = data_and_surrounding_pointers,
    blocks_to_tensor: Callable[[List[BlockType]], t.tensor] = blocks_to_tensor_truncate,
) -> Tuple[dgl.DGLGraph, Dict[Tuple[str, int], int]]:
    node_labels = node_labels or {}
    type_descriptor = {"kind": "struct", "name": user_type_name}
    nx_g = to_nx_graph(type_descriptor, symbol_encoder)

    relation_types = set(nx.get_edge_attributes(nx_g, "type").values())
    dgl_graph_data = {("slice", str(t), "slice"): _extract_adjacency(nx_g, t) for t in relation_types}
    dgl_graph_data, node_labels, dgl_node_ids = _map_out_nodes(dgl_graph_data, node_labels)

    dgl_graph = dgl.heterograph(dgl_graph_data)

    label_data = t.tensor([node_labels[n] for n, _ in sorted(dgl_node_ids.items(), key=lambda key_value: key_value[1])])
    dgl_graph.ndata["labels"] = label_data

    block_data = {}
    for node, data in nx_g.nodes.data():
        type_descriptor = json.loads(node.type_descriptor)
        blocks, _ = symbol_encoder.encode_type_descriptor(type_descriptor)  # Only need the blocks
        blocks = blocks[data["start"] : data["end"] + 1]
        tensor = blocks_to_tensor(blocks)
        tensor = t.flatten(tensor)
        block_data[dgl_node_ids[node]] = tensor

    block_data = t.stack([block_data[id] for id in range(max(block_data.keys()) + 1)])

    assert len(block_data) == len(label_data)

    dgl_graph.ndata["blocks"] = block_data
    return dgl_graph, node_labels
