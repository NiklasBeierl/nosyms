from collections import deque
import json
from typing import Callable, List, Tuple, Dict, NamedTuple
from warnings import warn
import dgl
import networkx as nx
import torch as t
from encoding import BlockType, VolatilitySymbolsEncoder
from encoding.block_types import blocks_to_tensor_truncate

# TODO: This import is not pretty, MemRelations and both data_and_surrounding_pointers functions should be grouped
# somewhere. Maybe together with the SandwichEncoder?
from encoding.memory_graph import MemRelations, _extract_adjacency


class NodeId(NamedTuple):
    type_descriptor: str
    chunk: int


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
            current_node = NodeId(curr_td_str, start)
            graph.add_node(current_node, start=start, end=end)
            if last_node:
                graph.add_edge(last_node, current_node, type=MemRelations.PRECEDS)
                graph.add_edge(current_node, last_node, type=MemRelations.FOLLOWS)
            if end in pointers:
                # This will also cover all the "start" of the next node
                if pointers[end] not in did_encode and pointers[end] not in to_encode:
                    to_encode.append(pointers[end])
                deferred_edges.append((NodeId(pointers[end], 0), current_node, MemRelations.BELOW_POINTED_TO))
            if start in pointers:
                deferred_edges.append((NodeId(pointers[start], 0), current_node, MemRelations.ABOVE_POINTED_TO))
            last_node = current_node
        did_encode.add(curr_td_str)
    for u, v, rel_type in deferred_edges:
        if u in graph:  # We might have encountered a key error above, in that case u can not be used.
            graph.add_edge(u, v, type=rel_type)
    return graph


def _map_out_nodes(
    dgl_graph_data: Dict, node_labels: Dict[NodeId, int]
) -> Tuple[Dict, Dict[NodeId, int], Dict[NodeId, int]]:
    new_dgl_data = {}
    local_node_ids: Dict[NodeId, int] = {}
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
        block_data[dgl_node_ids[node]] = tensor

    block_data = t.stack([block_data[id] for id in range(max(block_data.keys()) + 1)])

    assert len(block_data) == len(label_data)

    dgl_graph.ndata["blocks"] = block_data
    return dgl_graph, node_labels
