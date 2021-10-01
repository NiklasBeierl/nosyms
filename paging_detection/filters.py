from collections import defaultdict
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


def pml4_kernel_mapping_similarity(pages: Dict[int, PagingStructure]) -> Dict[int, float]:
    """
    Calculates a likelihood of actually being a pml4 for every candidate pml4 based on how many entries
    it shares with other pml4s in the "kernel" end of the address space.
    """
    candidate_pml4s = [offset for offset, page in pages.items() if PageTypes.PML4 in page.designations]

    entries: Dict[int, set] = {}

    entry_counts = defaultdict(lambda: 0)

    for page_offset in candidate_pml4s:
        page = pages[page_offset]
        entries[page_offset] = {
            (entry_offset, entry.value) for entry_offset, entry in page.entries.items() if entry_offset > 2048
        }
        for entry in entries[page_offset]:
            entry_counts[entry] += 1

    max_occurence = max(entry_counts.values())
    entry_scores = {entry: occurences / max_occurence for entry, occurences in entry_counts.items()}

    page_scores = {offset: sum(entry_scores[entry] for entry in entries) for offset, entries in entries.items()}
    max_score = max(page_scores.values())
    page_scores = {offset: score for offset, score in sorted(page_scores.items(), key=lambda tup: tup[1], reverse=True)}
    page_scores_normed = {offset: score / max_score for offset, score in page_scores.items()}

    return page_scores_normed


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

    # Discarding entries to page 0
    zero_entries = graph.in_degree["0"]
    page_zero = graph.nodes["0"]
    graph.remove_node("0")
    graph.add_node("0", **page_zero)  # Adding node back in to prevent keyerrors
    print(f"Removed {zero_entries} edges pointing to page 0.")

    no_zero = prune_designations(graph, pages)
    print(f"No-zero prune removed {initial_prune} designations.")

    # Discarding pages with invalid entries
    excluded = 0
    for node in graph.nodes.values():
        for page_type in PageTypes:
            if node[f"invalid_{page_type}"] > 0:
                excluded += 1
                node[str(page_type)] = False
    print(f"Removed {excluded} designations due to invalid entries.")
    pruned = prune_designations(graph, pages)
    print(f"Prune removed {pruned} designations.")

    # Discarding pages with OOB entries
    excluded = 0
    for node in graph.nodes.values():
        for page_type in PageTypes:
            if node[f"oob_{page_type}"] > 0:
                excluded += 1
                node[str(page_type)] = False
    print(f"Removed {excluded} designations due to OOB entries.")
    pruned = prune_designations(graph, pages)
    print(f"Prune removed {pruned} designations.")

    # Applying the "kernel mapping similarity" filter

    print("Transferring designations to snapshot data")
    for offset, node in graph.nodes.items():
        pages[int(offset)].designations = set(type for type in PageTypes if node[str(type)])

    pml4_scores = pml4_kernel_mapping_similarity(pages)
    removed = 0
    for page_offset, score in pml4_scores.items():
        if score < 0.8:
            removed += 1
            graph.nodes[str(page_offset)][str(PageTypes.PML4)] = False

    print(f"Removed {removed} PML4 designations based on kernel part similarities.")
    pruned = prune_designations(graph, pages)
    print(f"Prune removed {pruned} designations.")

    # Syncing and saving

    print("Transferring designations to snapshot data")
    for offset, node in graph.nodes.items():
        pages[int(offset)].designations = set(type for type in PageTypes if node[str(type)])

    print(f"Saving pages: {out_pages_path}")
    with open(out_pages_path, "w") as f:
        f.write(snapshot.json())

    print(f"Saving graph: {out_graph_path}")
    nx.readwrite.write_graphml(graph, out_graph_path)

print("Done")
