from typing import Dict, Iterable, Tuple, Union

import networkx as nx

from paging_detection import PageTypes, PagingStructure


def add_task_info(graph: nx.Graph, process_info: Iterable[Union[Tuple[int, int, str], Tuple[int, str]]]) -> nx.Graph:
    """
    Add process information to nodes in a paging structure graph.
    :param graph: Paging structure graph
    :param process_info: 3-tuples of either: kernel pml4 address, user pml4 address and a process name (comm)
    or 2- tuples of process pml4 address and process name (comm)
    :return: Copy of graph with task information added to nodes
    """
    graph = graph.copy()
    for proc in process_info:
        kernel_pml4, user_pml4, comm = proc if len(proc) == 3 else (None,) + proc
        graph.nodes[user_pml4]["comm"] = comm
        if kernel_pml4 is not None:
            graph.nodes[kernel_pml4]["comm"] = comm
            graph.nodes[kernel_pml4]["type"] = "kernel"
            graph.nodes[user_pml4]["type"] = "user"
            graph.add_edge(kernel_pml4, user_pml4)
    return graph


# Only listing the combinations I have witnessed so far.
DESIGNATION_COLORS = {
    frozenset({PageTypes.PML4}): "black",
    frozenset({PageTypes.PDP}): "red",
    frozenset({PageTypes.PD}): "green",
    frozenset({PageTypes.PT}): "blue",
    frozenset({PageTypes.PD, PageTypes.PT}): "cyan",
    frozenset({PageTypes.PDP, PageTypes.PT}): "magenta",
    frozenset({PageTypes.PDP, PageTypes.PD}): "yellow",
    frozenset({PageTypes.PDP, PageTypes.PD, PageTypes.PT}): "white",
}


def color_graph(graph: nx.Graph, pages: Dict[int, PagingStructure]) -> nx.Graph:
    """
    Color code paging structure graph according to paging structure designations.
    :param graph: Graph to add color coding to
    :param pages: Dict mapping physical address to paging structure
    :return: Copy of graph with color coding added to nodes
    """
    graph = graph.copy()
    for n in graph.nodes:
        graph.nodes[n]["color"] = DESIGNATION_COLORS[frozenset(pages[n].designations)]
    return graph
