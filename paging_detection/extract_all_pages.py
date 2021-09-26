import mmap
from typing import Dict

import networkx as nx

from paging_detection import PagingStructure, ReadableMem, PAGING_STRUCTURE_SIZE, Snapshot, max_page_addr
from paging_detection.pt_tests import maybepte

import numpy as np


def read_all(mem: ReadableMem, path: str) -> Dict[int, PagingStructure]:
    """
    Consider all 4kb pages in a mem a paging structure and create PagingStructure instances for every one of them.
    :param mem: Memory to read from
    :return: Dict mapping page address to instance of PagingStructure describing the underlying page
    """

    dumpfile = np.memmap(path, mode="r", dtype="uint64")
    maxpfn = len(dumpfile) // 512
    max_paddr = max_page_addr(len(mem))
    alt_pages = 0
    alt_no_oob = 0
    pages = {}

    print("Reading all pages.")
    last_progress = 0
    for offset, pageno in zip(range(0, len(mem), PAGING_STRUCTURE_SIZE), range(maxpfn)):
        if (prog := offset * 100 // len(mem)) != last_progress and prog % 5 == 0:
            print(f"{prog}%")
            last_progress = prog
        page = PagingStructure.from_mem(mem[offset : offset + PAGING_STRUCTURE_SIZE], designations=[])

        present, oob, bit7, not_present = maybepte(dumpfile, maxpfn, pageno, None)

        in_bounds = {o: e for o, e in page.entries.items() if e.target <= max_paddr and e.target != 0}

        oob_es = len(page.entries) - len(in_bounds)
        bit7_es = len({o: e for o, e in in_bounds.items() if e.value & (1 << 7)})
        present_es = len(in_bounds) - bit7_es

        if present != present_es or oob != oob_es or bit7 != bit7_es or not_present != (512 - len(page.entries)):
            print("Error!")

        if oob == 0:
            alt_no_oob += 1

        if (bit7 + oob) == 0:
            alt_pages += 1

        pages[offset] = page

    pages_no_oob = sum(1 for page in pages.values() if all(e.target <= max_paddr for e in page.entries.values()))
    pages_no_oob_noz = sum(
        1
        for page in pages.values()
        if all((e.target <= max_paddr) and (e.target != 0) and (not e.value & (1 << 7)) for e in page.entries.values())
    )

    assert pages_no_oob_noz == alt_pages

    return pages


def build_nx_graph(pages: Dict[int, PagingStructure], max_paddr: int) -> nx.MultiDiGraph:
    """
    Build a networkx graph representing pages and their (hypothetical) paging entries in a snapshot.
    :param pages: Dict mapping physical address to pages
    :return: The resulting graph
    """
    graph = nx.MultiDiGraph()
    for page_offset, page in pages.items():
        for entry_offset, entry in page.entries.items():
            if entry.target <= max_paddr:
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

    pages = read_all(mem, dump_path)

    max_paddr = max_page_addr(len(mem))

    print("Building nx graph.")
    full_graph = build_nx_graph(pages, max_paddr=max_paddr)

    print("Marking pages with oob entries.")
    for offset, page in pages.items():
        full_graph.nodes[offset]["has_oob_entries"] = any(entry.target > max_paddr for entry in page.entries.values())

    print(f"Saving graph: {out_graph_path}")
    nx.readwrite.write_graphml(full_graph, out_graph_path)

    # Keeping oob entries significantly increases the size of th pages json file.
    print("Removing out of bound entries from page data.")
    for page in pages.values():
        page.entries = {offset: entry for offset, entry in page.entries.items() if entry.target <= max_paddr}

    print(f"Saving pages: {out_pages_path}")
    snapshot = Snapshot(path=str(dump_path.resolve()), pages=pages, size=len(mem))
    with open(out_pages_path, "w") as f:
        f.write(snapshot.json())

    print("Done")
