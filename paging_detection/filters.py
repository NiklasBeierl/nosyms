from collections import defaultdict
from functools import cache
import json
from typing import Dict

import networkx as nx
import pandas as pd

from paging_detection import Snapshot, PageTypes, PagingStructure

_PAGE_TYPES_ORDERED = tuple(PageTypes)

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

    # print(f"Loading graph: {in_graph_path}")
    # graph = nx.read_graphml(in_graph_path, force_multigraph=True)

    print(f"Loading pages: {in_pages_path}")
    with open(in_pages_path) as f:
        snapshot = Snapshot.validate(json.load(f))

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

    print(f"Saving pages: {out_pages_path}")
    with open(out_pages_path, "w") as f:
        f.write(snapshot.json())

    # TODO: Sync graph and pages!
    # print(f"Saving graph: {out_graph_path}")
    # nx.readwrite.write_graphml(graph_with_types, out_graph_path)

print("Done")
