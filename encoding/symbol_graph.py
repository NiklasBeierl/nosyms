from collections import deque
import json
from typing import Callable, List, Tuple, NamedTuple
import dgl
import networkx as nx
import torch as t
from encoding import BlockType, VolatilitySymbolsEncoder
from encoding.block_types import blocks_to_tensor_truncate

# TODO: This import is not pretty, MemRelations and both data_and_surrounding_pointers functions should be grouped
# somewhere. Maybe together with the SandwichEncoder?
from encoding.memory_graph import MemRelations


def _pointer_delimited_chunks(blocks: List[BlockType]) -> List[Tuple[int, int]]:
    """
    Generate list of start, end interval from list of blocks, where every pointer block marks end of the previous and
    start of the next interval.
    :param blocks: List of blocks to generate intervals from.
    :return: List of start, end tuples with: result[i-i][1] == result[i][0]
    """
    result = []
    last = 0
    for offset, block in enumerate(blocks):
        if block == BlockType.Pointer:
            result.append((last, offset))
            last = offset
    if last != (len(blocks) - 1):
        result.append((last, len(blocks) - 1))
    return result


def data_and_surrounding_pointers(
    type_descriptor: "type_descriptor", memory_encoder: VolatilitySymbolsEncoder
) -> nx.DiGraph:
    graph = nx.DiGraph()
    to_encode = deque([json.dumps(type_descriptor)])
    did_encode = set()
    # TODO: Can nx handle edges with nodes that where not yet inserted?
    deferred_edges = []
    while to_encode:
        curr_td_str = to_encode.pop()
        try:
            blocks, pointers = memory_encoder.encode_type_descriptor(json.loads(curr_td_str))
        except KeyError:  # Apparently volatility symbols sometimes have type_descriptors without corresponding types.
            print(f"Undefined type encountered: {curr_td_str}")
            continue
        pointers = {offset: json.dumps(td) for offset, td in pointers.items()}
        if 0 in pointers and pointers[0] not in did_encode and pointers[0] not in to_encode:
            to_encode.append(pointers[0])

        last_node = None
        for start, end in _pointer_delimited_chunks(blocks):
            current_node = (curr_td_str, start)
            graph.add_node(current_node, start=start, end=end)
            if last_node:
                graph.add_edge(last_node, current_node, type=MemRelations.PRECEDS)
                graph.add_edge(current_node, last_node, type=MemRelations.FOLLOWS)
            if end in pointers:
                # This will also cover all the "start" of the next node
                if pointers[end] not in did_encode and pointers[end] not in to_encode:
                    to_encode.append(pointers[end])
                deferred_edges.append(((pointers[end]), current_node, MemRelations.BELOW_POINTED_TO))
            if start in pointers:
                deferred_edges.append(((pointers[start], 0), current_node, MemRelations.ABOVE_POINTED_TO))
            last_node = current_node
        did_encode.add(curr_td_str)
    for u, v, rel_type in deferred_edges:
        graph.add_edge(u, v, type=rel_type)
    return graph


def build_vol_symbols_graph(
    user_type_name: str,
    memory_encoder: VolatilitySymbolsEncoder,
    to_nx_graph: Callable[["type_descriptor", VolatilitySymbolsEncoder], nx.Graph] = data_and_surrounding_pointers,
    blocks_to_tensor: Callable[[List[BlockType]], t.tensor] = blocks_to_tensor_truncate,
) -> dgl.DGLGraph:
    type_descriptor = {"kind": "struct", "name": user_type_name}
    nx_graph = to_nx_graph(type_descriptor, memory_encoder)

    graph = nx_graph
    return graph
