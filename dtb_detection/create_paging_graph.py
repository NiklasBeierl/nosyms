from collections import defaultdict
from enum import Enum
import mmap
import struct
from typing import Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel, validator, ValidationError

from dtb_detection import ReadableMem, translate

PAGING_STRUCTURE_SIZE = 2 ** 12
PAGING_ENTRY_SIZE = 8


class InvalidEntry(ValidationError):
    ...


class InvalidPML4Entry(InvalidEntry):
    ...


class InvalidPDPEntry(InvalidEntry):
    ...


class GeneralEntry(BaseModel):
    value: int

    @property
    def present(self):
        return self.value & 1

    @property
    def target(self):
        return self.value & 0x001FFFFFFFFFF000

    @property
    def nx(self):
        return bool(self.value & 1 << 63)


class PML4Entry(GeneralEntry):
    @validator("value")
    def bit_8_and_7_mbz(cls, value):
        if (value & 1) and (value & (3 << 7)):
            raise InvalidPML4Entry
        return value

    @property
    def target_is_physical(self):
        return False


class PDPEntry(GeneralEntry):
    @validator("value")
    def one_gb_page_aligned(cls, value):
        if (value & 1) and (value & (1 << 7)) and (value & 0x1FFFF << 12):
            raise InvalidPDPEntry
        return value

    @property
    def target_is_physical(self):
        return self.value & (1 << 7)


class PDEntry(GeneralEntry):
    @validator("value")
    def two_mb_page_aligned(cls, value):
        if (value & 1) and (value & (1 << 7)) and (value & 0xFF << 12):
            raise InvalidPDPEntry
        return value

    @property
    def target_is_physical(self):
        return self.value & (1 << 7)


class PTEntry(GeneralEntry):
    @property
    def target_is_physical(self):
        return True


class PageTypes(Enum):
    PML4 = "PML4"
    PDP = "PDP"
    PD = "PD"
    PT = "PT"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


ENTRY_FOR_TABLE = {PageTypes.PML4: PML4Entry, PageTypes.PDP: PDPEntry, PageTypes.PD: PDEntry, PageTypes.PT: PTEntry}


def read_page(mem: ReadableMem, page_type: PageTypes) -> Dict[int, GeneralEntry]:
    assert len(mem) == PAGING_STRUCTURE_SIZE  # Fail fast

    table = {}
    for offset in range(0, PAGING_STRUCTURE_SIZE, PAGING_ENTRY_SIZE):
        value = struct.unpack("<Q", mem[offset : offset + 8])[0]
        try:
            entry = ENTRY_FOR_TABLE[page_type](value=value)
            if entry.present:
                table[offset] = entry
        except ValidationError:  # TODO: Why does catching InvalidEntry not work here?
            ...
    return table


def read_paging_structures(mem: ReadableMem, pgds: List[int]) -> Tuple[Dict, List]:
    # results: Dict[int, Dict[int, GeneralEntry]] = {}  # phy addr of page -> (Offset -> Entry)
    results = defaultdict(list)
    page_types: Dict[int, PageTypes] = {}
    warnings = []

    last_prog = 0
    for i, pml4_addr in enumerate(pgds):
        if (prog := 100 * i // len(pgds)) != last_prog and (prog % 5 == 0):
            print(f"{prog}% done")
            last_prog = prog

        addresses = [pml4_addr]  # , pml4_addr + PAGING_STRUCTURE_SIZE]
        for page_type in PageTypes:
            next_addresses = []  # Holds addresses of tables of the "next" table type by the end onf the iteration
            for addr in addresses:
                if addr in page_types:
                    if page_types[addr] != page_type:
                        warnings.append(
                            f"Expecting {page_type} table @{addr} but page is already known as {page_types[addr]}."
                        )
                    else:
                        continue  # Already read that
                if addr + PAGING_STRUCTURE_SIZE >= len(mem):
                    warnings.append(f"Expecting {page_type} table @{addr} but it would lie outside memory.")
                    continue  # Can't read outside memory
                page_mem = mem[addr : addr + PAGING_STRUCTURE_SIZE]
                table = read_page(page_mem, page_type)
                results[addr].append(table)
                page_types[addr] = page_type
                # PDEs and PDPEs can point to large pages.
                next_addresses += [entry.target for entry in table.values() if not entry.target_is_physical]
            addresses = next_addresses

    return results, warnings


if __name__ == "__main__":
    kernel_dtb = 39886848
    pml4ts = pd.read_csv("./nokaslr_pgds.csv")
    with open("../data_dump/nokaslr.raw", "rb") as f:
        mem = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        pml4ts["translated_pgd"] = [
            translate(mem, kernel_dtb, pgd) if pgd != -1 else -1 for pgd in pml4ts["active_mm->pgd"]
        ]

        phy_pgds = []
        for virt_pgd in pml4ts["active_mm->pgd"]:
            if virt_pgd != -1:
                """All top-level PAGE_TABLE_ISOLATION page tables are order-1 pages (8k-aligned and 8k in size).
                The kernel one is at the beginning 4k and the user one is in the last 4k.
                To switch between them, you just need to flip the 12th bit in their addresses."""
                # phy_pgds.append(translate(mem, kernel_dtb, (virt_pgd & ~(1 << 12))))  # Bit 12 cleared (Kernel space)
                phy_pgds.append(translate(mem, kernel_dtb, (virt_pgd | (1 << 12))))  # Bit 12 set (user space)

        results, warnings = read_paging_structures(mem, phy_pgds)
        average_entries = sum(len(page[0]) for page in results.values()) / len(results)
        total_used_pages = sum(len(page[0]) for page in results.values())
        conflict_mappings = [w for w in warnings if ("outside" not in w) and ("@0" not in w)]
    print("Done.")
