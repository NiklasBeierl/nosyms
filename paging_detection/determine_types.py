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


def determine_possible_types(graph: nx.MultiDiGraph, pages: Dict[int, PagingStructure]) -> nx.MultiDiGraph:
    """
    From the topology of a "page graph", infer the possible page_types for every page (node).
    Assumptions:
        - Only PML4s can have no inbound edges. (Higher level structures must exist in any hierarchy)
        - At least one valid entry under any assigned page_type
        - At least one entry all the way to a data page
    :param graph: Graph representing the pages.
    :param pages: Dict mapping physical address to a paging structure.
    :return: Graph with possible types of any page stored in its node data. (node[page_type] -> bool)
    """

    # graph = graph.copy() # Without this I am technically speaking mutating args, but the copy is costly.

    designations_avoided = 0
    for node in graph.nodes:
        page = pages[int(node)]
        # No dangling paging structures
        max_inbound = get_max_path(graph, node, max_len=len(PageTypes) - 1, direction="in")
        poss_types = set(PAGE_TYPES_ORDERED[: max_inbound + 1])

        max_outbound = get_max_path(graph, node, max_len=len(PageTypes) - 1, direction="out")

        if max_outbound == 0:  # Can only be a data page
            designations_avoided += len(poss_types)
            poss_types = set()
        elif max_outbound == 1:
            poss_types.discard(PageTypes.PML4)  # PML4s never directly point to data pages
            # PDP and PD can point to large pages, but there needs to be at least one qualifying entry
            for page_type in poss_types & {PageTypes.PDP, PageTypes.PD}:
                if not any(entry.target_is_data(page_type) for entry in page.entries.values()):
                    poss_types.discard(page_type)
                    designations_avoided += 1
        elif max_outbound == 2:
            suc_entries = [entry for suc in graph.successors(node) for entry in pages[int(suc)].entries.values()]
            # If none of the successors qualifies as a PDP pointing to a data page, the current page can't be a PML4
            if PageTypes.PML4 in poss_types and not any(entry.target_is_data(PageTypes.PDP) for entry in suc_entries):
                poss_types.discard(PageTypes.PML4)
                designations_avoided += 1
            # If none of the successors qualifies as a PD pointing to a data page, the current page can't be a PDP
            if PageTypes.PDP in poss_types and not any(entry.target_is_data(PageTypes.PD) for entry in suc_entries):
                poss_types.discard(PageTypes.PDP)
                designations_avoided += 1

        # At least one valid entry under any assigned page_type
        for page_type in poss_types & set(PAGE_TYPES_ORDERED[:-1]):  # PT entries are always valid
            if not any(entry.is_valid(page_type) for entry in page.entries.values()):
                poss_types.remove(page_type)
                designations_avoided += 1

        for t in PageTypes:
            graph.nodes[node][t] = t in poss_types

    return graph


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_files",
        help="Path to graphml file or the json with all pages in the snapshot. Other will be inferred.",
        type=pathlib.Path,
    )
    args = parser.parse_args()
    input_path = args.in_files
    if input_path.suffix not in {".json", ".graphml", ""}:
        raise ValueError("Invalid extension for input files path. Must be either .json, .graphml or no extension.")
    in_pages_path = input_path.with_suffix(".json")
    in_graph_path = input_path.with_suffix(".graphml")
    out_pages_path = input_path.with_stem(input_path.stem + "_with_types").with_suffix(".json")
    out_graph_path = out_pages_path.with_suffix(".graphml")

    print(f"Loading graph: {in_graph_path}")
    graph = nx.read_graphml(in_graph_path, force_multigraph=True)

    print(f"Loading pages: {in_pages_path}")
    with open(in_pages_path) as f:
        snapshot = Snapshot.validate(json.load(f))
    pages = snapshot.pages

    print("Determining possible types for all pages.")
    graph_with_types = determine_possible_types(graph, pages)

    print(f"Saving graph: {out_graph_path}")
    nx.readwrite.write_graphml(graph_with_types, out_graph_path)

    print("Transferring designations to snapshot data")
    for offset, node in graph_with_types.nodes.items():
        pages[int(offset)].designations = {t for t in PageTypes if node[t]}

    print(f"Saving pages: {out_pages_path}")
    with open(out_pages_path, "w") as f:
        f.write(snapshot.json())

print("Done")
