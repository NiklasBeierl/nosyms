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


DUMP_NAME = "../data_dump/nokaslr.raw"

if __name__ == "__main__":
    with open(DUMP_NAME, "rb") as f:
        mem = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    pages = read_all(mem)

    print("Removing out of bound entries.")
    for page in pages.values():
        page.entries = {offset: entry for offset, entry in page.entries.items() if entry.target < len(mem)}

    print("Saving pages.")
    snapshot = Snapshot(path=DUMP_NAME, pages=pages, size=len(mem))
    with open("../data_dump/all_pages.json", "w") as f:
        f.write(snapshot.json())

    print("Building nx graph.")
    full_graph = build_nx_graph(pages)

    print("Saving graph.")
    nx.readwrite.write_graphml(full_graph, "../data_dump/all_pages.graphml")

    print("Done")
