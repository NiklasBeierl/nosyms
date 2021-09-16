import mmap
from typing import Dict

import networkx as nx

from paging_detection import PagingStructure, ReadableMem, PAGING_STRUCTURE_SIZE, Snapshot


def read_all(mem: ReadableMem) -> Dict[int, PagingStructure]:
    """
    Consider all 4kb pages in a mem a paging structure and create PagingStructure instances for every one of them.
    :param mem: Memory to read from
    :return: Dict mapping page address to instance of PagingStructure describing the underlying page
    """
    pages = {}

    print("Reading all pages.")
    last_progress = 0
    for offset in range(0, len(mem), PAGING_STRUCTURE_SIZE):
        if (prog := offset * 100 // len(mem)) != last_progress and prog % 5 == 0:
            print(f"{prog}%")
            last_progress = prog

        pages[offset] = PagingStructure.from_mem(mem[offset : offset + PAGING_STRUCTURE_SIZE], designations=[])
    return pages


def build_nx_graph(pages: Dict[int, PagingStructure]) -> nx.MultiDiGraph:
    """
    Build a networkx graph representing pages and their (hypothetical) paging entries in a snapshot.
    :param pages: Dict mapping physical address to pages
    :return: The resulting graph
    """
    graph = nx.MultiDiGraph()
    for page_offset, page in pages.items():
        for entry_offset, entry in page.entries.items():
            graph.add_edge(page_offset, entry.target, entry_offset)

    graph.add_nodes_from(pages.keys())  # Adds "disconnected" pages. Mostly to avoid key errors.

    return graph


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_file",
        help="Path to snapshot. Output files will have the same name with .json and .graphml as suffix.",
        type=pathlib.Path,
    )
    args = parser.parse_args()
    dump_path = args.in_file
    if dump_path.suffix in {".json", ".graphml"}:
        raise ValueError(f"Snapshot has {dump_path.suffix} as extension and would be overwritten by outputs.")
    out_pages_path = dump_path.with_stem(dump_path.stem + "_all_pages").with_suffix(".json")
    out_graph_path = out_pages_path.with_suffix(".graphml")

    with open(dump_path, "rb") as f:
        mem = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    pages = read_all(mem)

    print("Removing out of bound entries.")
    for page in pages.values():
        page.entries = {offset: entry for offset, entry in page.entries.items() if entry.target < len(mem)}

    print(f"Saving pages: {out_pages_path}")
    snapshot = Snapshot(path=str(dump_path.resolve()), pages=pages, size=len(mem))
    with open(out_pages_path, "w") as f:
        f.write(snapshot.json())

    print("Building nx graph.")
    full_graph = build_nx_graph(pages)

    print(f"Saving graph: {out_graph_path}")
    nx.readwrite.write_graphml(full_graph, out_graph_path)

    print("Done")
