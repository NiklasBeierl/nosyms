from typing import Dict, Iterable, Tuple

import networkx as nx

from dtb_detection import PageTypes, PagingStructure


def build_nx_graph(pages: Dict[int, PagingStructure], mem_size: int, include_phyiscal: bool = False) -> nx.DiGraph:
    """
    Build a networkx graph representing the paging structures in a snapshot.
    :param pages: Dict mapping physical address to paging structure.
    :param mem_size: Size of the memory snapshot, pages "outside" the physical memory will be ignored.
    :return:
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(pages.keys())
    for offset, page in pages.items():
        for designation in page.designations:
            for entry in page.entries.values():
                if entry.target < mem_size and (include_phyiscal or not entry.target_is_physical(designation)):
                    graph.add_edge(offset, entry.target)

    for offset, page in pages.items():
        graph.nodes[offset].update({t: (t in page.designations) for t in PageTypes})

    return graph


def add_task_info(graph: nx.Graph, process_info: Iterable[Tuple[str, int, int]]) -> nx.Graph:
    """
    Add process information to nodes in a paging structure graph.
    :param graph: Paging structure graph
    :param process_info: Tuples of kernel pml4 address, user pml4 address and a process name (comm)
    :return: Copy of graph with task information added to nodes
    """
    graph = graph.copy()
    for kernel_pml4, user_pml4, comm in process_info:
        graph.nodes[kernel_pml4]["comm"] = graph.nodes[user_pml4]["comm"] = comm
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
