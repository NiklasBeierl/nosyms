from collections import defaultdict, Counter
import mmap
from typing import Dict, List, DefaultDict

import pandas as pd
import networkx as nx

from dtb_detection import ReadableMem, translate, PagingStructure, PageTypes, PAGING_STRUCTURE_SIZE
from dtb_detection.graphs import build_nx_graph, color_graph, add_task_info


def read_paging_structures(mem: ReadableMem, pgds: List[int]) -> Dict[int, PagingStructure]:
    pages: Dict[int, PagingStructure] = {}
    print("Reading known paging data.")
    last_progress = 0
    for i, pml4_addr in enumerate(pgds):
        if (progress := 100 * i // len(pgds)) != last_progress and (progress % 5 == 0):
            print(f"{progress}%")
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


def get_node_features(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Create a pandas dataframe with some useful stats for every node in a paging structures graph.
    """
    df = pd.DataFrame(data=[graph.nodes[node] for node in graph.nodes], index=graph.nodes)
    r_graph = graph.reverse()
    df["longest_inbound_path"] = [len(nx.dfs_successors(r_graph, source=node)) for node in r_graph.nodes]
    idx, deg = tuple(zip(*graph.in_degree))
    df["in_degree"] = pd.Series(data=deg, index=idx)
    idx, deg = tuple(zip(*graph.out_degree))
    # Note: Out degree will be 0 for pages in "training" data and 1 in "target" data
    df["out_degree"] = pd.Series(data=deg, index=idx)
    return df


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
    graph = add_task_info(graph, task_info[["phy_kernel_pml4", "phy_user_pml4", "COMM"]].itertuples(index=False))

    node_data = get_node_features(graph)

    graph = color_graph(graph, pages)
    nx.readwrite.write_graphml(graph, "./page-structures.graphml")

    print("Done.")
