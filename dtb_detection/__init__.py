import functools
import struct
from mmap import mmap
from typing import Tuple, Mapping, Union

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
