from collections import defaultdict, Counter
import mmap
from typing import Dict, List, Iterable, DefaultDict, Tuple

import pandas as pd
import networkx as nx

from dtb_detection import ReadableMem, translate, PagingStructure, PageTypes, PAGING_STRUCTURE_SIZE


def read_paging_structures(mem: ReadableMem, pgds: List[int]) -> Dict[int, PagingStructure]:
    pages: Dict[int, PagingStructure] = {}

    last_progress = 0
    for i, pml4_addr in enumerate(pgds):
        if (progress := 100 * i // len(pgds)) != last_progress and (progress % 5 == 0):
            print(f"{progress}% done")
            last_progress = progress

        addresses = {pml4_addr}
        for page_type in PageTypes:
            next_addresses = set()  # Holds addresses of tables of the "next" table type by the end onf the iteration
            for addr in addresses:
                if addr in pages:  # Table already parsed
                    table = pages[addr]
                    if page_type in table.designations:
                        continue  # Already considered this table as page_type, nothing left to do
                    # Not continue-ing here since we now need to consider all entries given the new type designation
                    table.designations.add(page_type)
                else:
                    page_mem = mem[addr : addr + PAGING_STRUCTURE_SIZE]
                    table = PagingStructure.from_mem(mem=page_mem, designations={page_type})
                    pages[addr] = table
                next_addresses |= {
                    entry.target
                    for entry in table.entries.values()
                    # PDEs and PDPEs can point to large pages, we do not want to confuse those for paging structures
                    if not entry.target_is_physical(page_type) and entry.target < len(mem)
                }
            addresses = next_addresses

    return pages


def get_mapped_pages(pages: Dict[int, PagingStructure]) -> DefaultDict[int, bool]:
    is_mapped = defaultdict(lambda: False)
    for page in pages.values():
        for entry in page.entries.values():
            for designation in page.designations:
                if entry.target_is_physical(designation):
                    is_mapped[entry.target] = True
    return is_mapped


def build_nx_graph(pages: Dict[int, PagingStructure], mem_size: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(pages.keys())
    # graph.add_nodes_from(pages.keys())
    for offset, page in pages.items():
        for designation in page.designations:
            for entry in page.entries.values():
                if entry.target < mem_size and not entry.target_is_physical(designation):
                    graph.add_edge(offset, entry.target)

    for n in graph.nodes:
        desigs = pages[n].designations
        graph.nodes[n].update({str(t): (t in desigs) for t in PageTypes})

    return graph


def add_task_info(graph, info: Iterable[Tuple[str, int, int]]):
    for k, u, comm in info:
        graph.nodes[k]["comm"] = graph.nodes[u]["comm"] = comm
        graph.nodes[k]["type"] = "kernel"
        graph.nodes[u]["type"] = "user"
        graph.add_edge(k, u)


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


def color_graph(graph: nx.Graph, pages: Dict[int, PagingStructure]):
    for n in graph.nodes:
        graph.nodes[n]["color"] = DESIGNATION_COLORS[frozenset(pages[n].designations)]


if __name__ == "__main__":
    kernel_dtb = 39886848
    task_info = pd.read_csv("./nokaslr_pgds.csv")
    with open("../data_dump/nokaslr.raw", "rb") as f:
        mem = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    # https://elixir.bootlin.com/linux/latest/source/arch/x86/include/asm/pgtable.h#L1196
    # All top-level PAGE_TABLE_ISOLATION page tables are order-1 pages (8k-aligned and 8k in size).
    # The kernel one is at the beginning 4k and the user one is in the last 4k.
    # To switch between them, you just need to flip the 12th bit in their addresses.
    task_info["phy_kernel_pml4"] = [  # Bit 12 cleared (kernel space)
        translate(mem, kernel_dtb, (pgd & ~(1 << 12))) if pgd != -1 else -1 for pgd in task_info["active_mm->pgd"]
    ]
    task_info["phy_user_pml4"] = [  # Bit 12 set (user space)
        translate(mem, kernel_dtb, (pgd | (1 << 12))) if pgd != -1 else -1 for pgd in task_info["active_mm->pgd"]
    ]
    task_info = task_info[task_info["phy_user_pml4"] != -1]

    phy_pgds = []
    for kernel, user in task_info[["phy_kernel_pml4", "phy_user_pml4"]].itertuples(index=False):
        if user != -1:
            phy_pgds.extend([kernel, user])

    pages = read_paging_structures(mem, phy_pgds)

    is_mapped = get_mapped_pages(pages)
    total_mapped = sum(mapped and addr < len(mem) for addr, mapped in is_mapped.items())
    # Approximate b.c. of large pages
    app_mapped_mem_perc = total_mapped / (len(mem) / PAGING_STRUCTURE_SIZE)

    types_summary = Counter((is_mapped[addr], *page.designations) for addr, page in pages.items())
    ambiguous_pages = sum(occ for desigs, occ in types_summary.items() if len(desigs) > 2)

    out_of_bounds = defaultdict(set)
    for address, page in pages.items():
        for offset, entry in page.out_of_bounds_entries(len(mem)).items():
            out_of_bounds[address].add((offset, entry.target))

    out_of_bounds_entries = sum(len(entries) for entries in out_of_bounds.values())
    out_of_bound_pages = sum(len(set(target for _, target in entries)) for entries in out_of_bounds.values())

    graph = build_nx_graph(pages, len(mem))
    add_task_info(graph, task_info[["phy_kernel_pml4", "phy_user_pml4", "COMM"]].itertuples(index=False))
    color_graph(graph, pages)
    nx.readwrite.write_graphml(graph, "./page-structures.graphml")

    print("Done.")
