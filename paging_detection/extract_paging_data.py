import mmap
from typing import Dict

import networkx as nx

from paging_detection import PagingStructure, ReadableMem, PAGING_STRUCTURE_SIZE, PageTypes, Snapshot
from paging_detection.graphs import build_nx_graph


# TODO: Parallelize?
def read_all(mem: ReadableMem) -> Dict[int, PagingStructure]:
    pages = {}
    last_progress = 0
    for offset in range(0, len(mem), PAGING_STRUCTURE_SIZE):
        if (prog := offset * 100 // len(mem)) != last_progress and prog % 5 == 0:
            print(f"{prog}%")
            last_progress = prog
        # Considering all possible designations
        pages[offset] = PagingStructure.from_mem(mem[offset : offset + PAGING_STRUCTURE_SIZE], designations=PageTypes)
    return pages


TYPES_ORDERED = tuple(PageTypes)[::-1]


def determine_possible_designations(graph: nx.DiGraph) -> nx.DiGraph:
    graph = graph.copy()
    for node in graph.nodes:
        max_outbound = len(nx.dfs_successors(graph, source=node, depth_limit=len(TYPES_ORDERED)))
        node_designations = TYPES_ORDERED[0:max_outbound]
        for t in PageTypes:
            graph.nodes[node][t] = t in node_designations
    return graph


DUMP_NAME = "../data_dump/nokaslr.raw"

if __name__ == "__main__":
    with open(DUMP_NAME, "rb") as f:
        mem = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    print("Reading in all pages")
    pages = read_all(mem)

    print("Removing out of bound entries.")
    for page in pages.values():
        page.entries = {offset: entry for offset, entry in page.entries.items() if entry.target < len(mem)}

    print("Building nx graph.")
    full_graph = build_nx_graph(pages, mem_size=len(mem), include_phyiscal=True)

    print("Determining designations.")
    full_graph = determine_possible_designations(full_graph)

    for offset, node in full_graph.nodes.items():
        pages[offset].designations = {t for t in PageTypes if node[t]}

    print("Saving graph.")
    nx.readwrite.write_graphml(full_graph, "./all_pages.graphml")

    print("Saving pages.")
    snapshot_data = Snapshot(path=DUMP_NAME, pages=pages, size=len(mem))
    with open("./all_pages.json", "w") as f:
        f.write(snapshot_data.json())

    print("Done")
