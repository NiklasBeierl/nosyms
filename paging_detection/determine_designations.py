import json
from typing import Dict, Literal, Union

import networkx as nx

from paging_detection import Snapshot, PageTypes, PagingStructure

PAGE_TYPES_ORDERED = tuple(PageTypes)


def get_max_path(graph: nx.MultiDiGraph, node, max_len: int, direction: Union[Literal["in"], Literal["out"]]) -> int:
    """
    Given a graph and one of its nodes, calculate the maximum length of any out / inbound paths, up to max_len.
    If there is an in / outbound cycle shorter than max_len the result will be max_len.
    :param graph: The graph.
    :param node: Id of start node in the graph.
    :param max_len: Maximum length to consider.
    :param direction: Whether to look at inbound or outbound paths.
    :return: The maximum path length considered.
    """
    path_len = 0
    next_nodes = {node}
    next_func = graph.successors if direction == "out" else graph.predecessors

    # TODO: Can abort sooner when there is any cycle with len < max_len
    # Since I currently use this with max_len == 3, it doesn't matter much.
    while (next_nodes := {suc for node in next_nodes for suc in next_func(node)}) and path_len < max_len:
        path_len += 1

    return path_len


def determine_possible_designations(graph: nx.MultiDiGraph, pages: Dict[int, PagingStructure]) -> nx.MultiDiGraph:
    """
    From the topology of a "page graph", infer the possible designation for every page (node).
    Assumptions:
        - Only PML4s can have no inbound edges. (Higher level structures must exist in any hierarchy)
        - At least one valid entry under any assigned designation
        - At least one entry all the way to a data page
    :param graph: Graph representing the pages.
    :param pages: Dict mapping physical addresses to a paging structure.
    :return: Graph with possible designations of any page stored in its node data. (node[page_type] -> bool)
    """
    # graph = graph.copy() # Without this I am technically speaking mutating args, but the copy is costly.
    designations_avoided = 0
    for node in graph.nodes:
        page = pages[int(node)]
        # No dangling paging structures
        max_inbound = get_max_path(graph, node, max_len=len(PageTypes) - 1, direction="in")
        possible_designations = set(PAGE_TYPES_ORDERED[: max_inbound + 1])

        max_outbound = get_max_path(graph, node, max_len=len(PageTypes) - 1, direction="out")

        if max_outbound == 0:  # Can only be physical page
            designations_avoided += len(possible_designations)
            possible_designations = set()
        elif max_outbound == 1:
            possible_designations.discard(PageTypes.PML4)  # PML4s never directly point to physical pages
            # PDP and PD can point to large pages, but there needs to be at least one qualifying entry
            for page_type in possible_designations & {PageTypes.PDP, PageTypes.PD}:
                if not any(entry.target_is_physical(page_type) for entry in page.entries.values()):
                    possible_designations.discard(page_type)
                    designations_avoided += 1
        elif max_outbound == 2 and PageTypes.PDP in possible_designations:
            children_entries = [entry for suc in graph.successors(node) for entry in pages[int(suc)].entries.values()]
            # If none of the successors qualifies as a PD pointing to a physical page, the current page can't be a PDP
            if not any(entry.target_is_physical(PageTypes.PD) for entry in children_entries):
                possible_designations.discard(PageTypes.PDP)
                designations_avoided += 1

        # At least one valid entry under any assigned designation
        for page_type in possible_designations & set(PAGE_TYPES_ORDERED[:-1]):  # PT entries are always valid
            if not any(entry.is_valid(page_type) for entry in page.entries.values()):
                possible_designations.remove(page_type)
                designations_avoided += 1

        for t in PageTypes:
            graph.nodes[node][t] = t in possible_designations

    return graph


if __name__ == "__main__":
    graph = nx.read_graphml("../data_dump/all_pages.graphml", force_multigraph=True)
    print("Loaded graph.")

    with open("../data_dump/all_pages.json") as f:
        snapshot = Snapshot.validate(json.load(f))
    pages = snapshot.pages
    print("Loaded pages.")

    print("Determining designations.")
    graph_with_designations = determine_possible_designations(graph, pages)

    print("Saving graph.")
    nx.readwrite.write_graphml(graph_with_designations, "../data_dump/all_pages_with_designations.graphml")

    print("Transferring designations to snapshot data")
    for offset, node in graph_with_designations.nodes.items():
        pages[int(offset)].designations = {t for t in PageTypes if node[t]}

    print("Saving pages.")
    with open("../data_dump/all_pages_with_designations.json", "w") as f:
        f.write(snapshot.json())

print("Done")
