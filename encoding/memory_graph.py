import dgl
import networkx as nx
import torch as t
from encoding import BlockType, MemoryEncoder
from encoding.block_types import blocks_to_tensor_truncate
from enum import Enum
from typing import Tuple, List, Iterable, Callable, NamedTuple


class Pointer:
    offset: int  # Where is the pointer?
    target: int  # Where does it point to?


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


def data_and_surrounding_pointers(pointers: List[Pointer]) -> nx.Graph:
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
        nodes.append(MemSlice(i, start_offset, p.offset))
        p_offset_to_node_id[p.offset] = i
        graph.add_node(i, start=start_offset, end=p.offset)  # Inclusive end
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
    nx_g = to_nx_graph(pointers)
    relation_types = set(nx.get_edge_attributes(nx_g, "type").values())
    dgl_graph_data = {("slice", str(t), "slice"): _extract_adjacency(nx_g, t) for t in relation_types}
    dgl_graph = dgl.heterograph(dgl_graph_data)
    ndata = {id: memory_encoder.encode(**data) for id, data in nx_g.nodes.data()}
    ndata = {id: blocks_to_tensor(blocks_list) for id, blocks_list in ndata.items()}
    default = t.zeros(next(iter(ndata.values())).shape)
    ndata = t.stack([ndata.get(id, default) for id in range(max(ndata.keys()) + 1)])
    dgl_graph.ndata["blocks"] = ndata
    return dgl_graph
