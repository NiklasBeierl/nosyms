import json
from typing import Dict

import networkx as nx

from paging_detection import Snapshot, PageTypes, PagingStructure, next_type, prev_type, PAGE_TYPES_ORDERED


def prune_designations(graph: nx.MultiDiGraph, pages: Dict[int, PagingStructure]) -> int:
    need_check = graph.nodes
    removed = 0
    while need_check:
        print(f"{len(need_check)} need checking.")
        next_need_check = set()
        for p_offset in need_check:
            node = graph.nodes[p_offset]
            page = pages[int(p_offset)]
            modified = False
            used_entries = [page.entries[int(e_offset)] for _, _, e_offset in graph.out_edges(p_offset, keys=True)]
            designations = set(type for type in PageTypes if node[str(type)])
            if PageTypes.PT in designations and not used_entries:  # Needs to point somewhere
                node[str(PageTypes.PT)] = False
                modified = True
            for type in designations.intersection(PAGE_TYPES_ORDERED[:-1]):
                if not (
                    # Points to something with the "next type"
                    any(graph.nodes[succ][str(next_type(type))] for succ in graph.successors(p_offset))
                    # Points to data
                    or any(entry.target_is_data(type) for entry in used_entries)
                ):
                    node[str(type)] = False
                    removed += 1
                    modified = True
            for type in designations.intersection(PAGE_TYPES_ORDERED[1:]):
                # Has a matching predecessor
                if not any(graph.nodes[pred][str(prev_type(type))] for pred in graph.predecessors(p_offset)):
                    node[str(type)] = False
                    removed += 1
                    modified = True
            if modified:
                next_need_check.update(graph.neighbors(p_offset))
        need_check = next_need_check
    return removed


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
    out_pages_path = input_path.with_stem(input_path.stem + "_filtered").with_suffix(".json")
    out_graph_path = out_pages_path.with_suffix(".graphml")

    print(f"Loading graph: {in_graph_path}")
    graph = nx.read_graphml(in_graph_path, force_multigraph=True)

    print(f"Loading pages: {in_pages_path}")
    with open(in_pages_path) as f:
        snapshot = Snapshot.validate(json.load(f))

    pages = snapshot.pages

    initial_prune = prune_designations(graph, pages)
    print(f"Initial prune removed {initial_prune} designations.")

    zero_entries = graph.in_degree["0"]
    page_zero = graph.nodes["0"]
    graph.remove_node("0")
    graph.add_node("0", **page_zero)  # Adding node back in to prevent keyerrors
    print(f"Removed {zero_entries} edges pointing to page 0.")

    no_zero = prune_designations(graph, pages)
    print(f"No-zero prune removed {initial_prune} designations.")

    excluded = 0
    for node in graph.nodes.values():
        for page_type in PageTypes:
            if node[f"invalid_{page_type}"] > 0:
                excluded += 1
                node[str(page_type)] = False
    print(f"Removed {excluded} designations due to invalid entries.")
    pruned = prune_designations(graph, pages)
    print(f"Prune removed {pruned} designations.")

    excluded = 0
    for node in graph.nodes.values():
        for page_type in PageTypes:
            if node[f"oob_{page_type}"] > 0:
                excluded += 1
                node[str(page_type)] = False
    print(f"Removed {excluded} designations due to OOB entries.")
    pruned = prune_designations(graph, pages)
    print(f"Prune removed {pruned} designations.")

    print("Transferring designations to snapshot data")
    for offset, node in graph.nodes.items():
        pages[int(offset)].designations = set(type for type in PageTypes if node[str(type)])

    """ 
    kernel_only_pages = {offset: page[2048:] for offset, page in snapshot.pages.items()}

    kernel_ends = defaultdict(list)

    for offset, page in kernel_only_pages.items():
        if PageTypes.PML4 in page.designations:
            kernel_ends[frozenset((offset, entry.value) for offset, entry in page.entries.items())].append(offset)

    khead_counts = {khead: len(pages) for khead, pages in kernel_ends.items()}
    khead_relevant = {khead: len(pages) for khead, pages in kernel_ends.items() if len(pages) >= 5}

    removed = sum(len(pages) for khead, pages in kernel_ends.items() if len(pages) < 5)
    for end, pages in kernel_ends.items():
        if len(pages) < 5:
            for po in pages:
                snapshot.pages[po].designations.discard(PageTypes.PML4)
    print(f"Filtered {removed} PML4s with less than 5 siblings.")

    PML4_ENTRY_LIMIT = 10
    filtered_large_pml4s = 0
    for page in snapshot.pages.values():
        if PageTypes.PML4 in page.designations and len(page.valid_pml4es) > PML4_ENTRY_LIMIT:
            page.designations.remove(PageTypes.PML4)
            filtered_large_pml4s += 1
    print(f"Filtered {filtered_large_pml4s} PML4s with more than {PML4_ENTRY_LIMIT} entries.")
    """

    print(f"Saving pages: {out_pages_path}")
    with open(out_pages_path, "w") as f:
        f.write(snapshot.json())

    print(f"Saving graph: {out_graph_path}")
    nx.readwrite.write_graphml(graph, out_graph_path)

print("Done")
