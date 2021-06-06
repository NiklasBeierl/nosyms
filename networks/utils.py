from typing import List, Optional, Tuple
import torch as t
from torch.nn.functional import one_hot
import dgl


def one_hot_with_neutral(tensor: t.tensor, neutral_class: int = 0, **kwargs):
    """
    One-hot encode a tensor while removing all scalars corresponding to a specified `neutral_class`.
    :param tensor: tensor to one-hot-encode.
    :param neutral_class: class whos corresponding scalars will be removed.
    :param kwargs: passed to torch.nn.functional.one_hot.
    :return: one hot encoded tensor one hot encoded tensor.
    """
    result = one_hot(tensor, **kwargs)
    classes = result.shape[-1]
    if neutral_class >= classes:
        raise ValueError("Neutral class not in classes.")
    result = result[:, :, [i for i in range(result.shape[-1]) if i != neutral_class]]
    return result


def add_self_loops(
    graph: dgl.DGLHeteroGraph,
    etype: Optional[Tuple[str, str, str]] = None,
    etypes: Optional[List[Tuple[str, str, str]]] = None,
) -> dgl.DGLHeteroGraph:
    """
    Add self loops to a graph in four possible configurations:
    If no kwargs are passed: Add self loops to all existing etypes in the graph.
    If `etype` is passed and exists in the graph add self loops for that etype.
    If `etype` is passed and does not exist the graph create a new graph with etype as an additional relationship
    which only contains self loops. (A new graph is created due to limitations of dgl)
    If `etypes` is passed, all the contained etypes need to exist in the graph and self loops will be added for all of
    them.
    :param graph: graph to add self loops to.
    :param etype: canonical etype to add self loops for. Or etype to add to the graph, only containing self loops
    :param etypes: canonical etypes to add self loops for
    :return: graph with self loops.
    """
    if etype is not None and etypes is not None:
        raise ValueError("You may only use etype or etypes, not both.")

    if etypes is None and etype is None:  # Add loops to all existing etypes
        for etype in graph.canonical_etypes:
            graph = dgl.add_self_loop(graph, etype=etype)
    elif etype and etype not in graph.canonical_etypes:  # New relationship
        new_graph = {etype: graph.adj(etype=etype, scipy_fmt="coo").nonzero() for etype in graph.canonical_etypes}
        new_graph[etype] = (range(graph.num_nodes()), range(graph.num_nodes()))
        new_graph = dgl.heterograph(new_graph)
        for k, v in graph.ndata.items():
            new_graph.ndata[k] = v
        graph = new_graph
    else:  # list of or individual etypes
        etypes = etypes if etypes else [etype]
        for etype in etypes:
            graph = dgl.add_self_loop(graph, etype=etype)

    return graph
