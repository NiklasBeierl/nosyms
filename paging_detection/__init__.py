from enum import Enum
import functools
from mmap import mmap
import struct
from typing import Tuple, Mapping, Union, Iterable, Dict, Set

from pydantic import BaseModel, Field

ReadableMem = Union[Mapping[slice, bytes], mmap]


class InvalidAddressException(Exception):
    ...


@functools.lru_cache(maxsize=None)
def dir2base(layer: ReadableMem, table_addr: int, index: int) -> Tuple[int, int]:
    entry_addr = table_addr + 8 * index
    entry = struct.unpack("<Q", layer[entry_addr : entry_addr + 8])[0]

    if entry & 1 == 0:  # not present
        raise InvalidAddressException("dir2base", table_addr, "Page not present")

    next_page = entry & 0x001FFFFFFFFFF000
    fields = entry & 0xFFF
    return next_page, fields


@functools.lru_cache(maxsize=None)
def translate(layer: ReadableMem, dtb: int, vaddr: int) -> int:
    (l4, f4) = dir2base(layer, dtb, (vaddr >> 39) & 0x1FF)
    (l3, f3) = dir2base(layer, l4, (vaddr >> 30) & 0x1FF)
    if f3 & 0x80:
        return l3 + (vaddr & ((1 << 30) - 1))
    (l2, f2) = dir2base(layer, l3, (vaddr >> 21) & 0x1FF)
    if f2 & 0x80:
        return l2 + (vaddr & ((1 << 21) - 1))
    (l1, f1) = dir2base(layer, l2, (vaddr >> 12) & 0x1FF)
    paddr = l1 + (vaddr & ((1 << 12) - 1))
    return paddr


PAGING_STRUCTURE_SIZE = 2 ** 12
PAGING_ENTRY_SIZE = 8


class PageTypes(Enum):
    PML4 = "PML4"
    PDP = "PDP"
    PD = "PD"
    PT = "PT"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


class PagingEntry(BaseModel):
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

    @property
    def valid_pml4e(self):
        # If bit 0 is set, bits 8 and 7 mbz
        return not ((self.value & 1) and (self.value & (3 << 7)))

    @property
    def valid_pdpe(self):
        # If bit 0 is set (present), bit 7 mbz or bits 13 through 29 mbz (1GiB aligned page addr)
        return not ((self.value & 1) and (self.value & (1 << 7)) and (self.value & 0x1FFFF << 12))

    @property
    def valid_pde(self):
        # If bit 0 is set (present), bit 7 mbz or bits 13 through 20 mbz (2MiB aligned page addr)
        return not ((self.value & 1) and (self.value & (1 << 7)) and (self.value & 0xFF << 12))

    # There is no valid_pt, because page tables have no invariants.

    def target_is_physical(self, assumed_type: PageTypes):
        if assumed_type == PageTypes.PML4:
            return False
        elif assumed_type == PageTypes.PDP:
            return self.valid_pdpe and (self.value & (1 << 7))
        elif assumed_type == PageTypes.PD:
            return self.valid_pde and (self.value & (1 << 7))
        elif assumed_type == PageTypes.PT:
            return True


class PagingStructure(BaseModel):
    entries: Dict[int, PagingEntry]
    designations: Set[PageTypes] = Field(default_factory=set)

    @property
    def valid_pml4es(self):
        return {offset: entry for offset, entry in self.entries.items() if entry.valid_pml4e}

    @property
    def valid_pdpes(self):
        return {offset: entry for offset, entry in self.entries.items() if entry.valid_pdpe}

    @property
    def valid_pdes(self):
        return {offset: entry for offset, entry in self.entries.items() if entry.valid_pde}

    @property
    def valid_ptes(self):
        return self.entries

    def out_of_bounds_entries(self, mem_size: int) -> Dict[int, PagingEntry]:
        return {offset: entry for offset, entry in self.entries.items() if entry.present and entry.target > mem_size}

    @classmethod
    def from_mem(cls, mem: ReadableMem, designations: Iterable[PageTypes]) -> "PagingStructure":
        assert len(mem) == PAGING_STRUCTURE_SIZE
        entries = {}
        for offset in range(0, PAGING_STRUCTURE_SIZE, PAGING_ENTRY_SIZE):
            value = struct.unpack("<Q", mem[offset : offset + 8])[0]
            if value & 1:  # Only add present entries
                entries[offset] = PagingEntry(value=value)
        return cls(entries=entries, designations=set(designations))


class Snapshot(BaseModel):
    path: str
    pages: Dict[int, PagingStructure]
    size: int
