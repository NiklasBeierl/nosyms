import struct
import csv
import re
import functools
from typing import List, Generator
from volatility3.framework import renderers, interfaces, layers
from volatility3.framework.configuration import requirements
from volatility3.framework.interfaces import plugins
from volatility3.framework.interfaces.configuration import path_join as join
from volatility3.framework.exceptions import InvalidAddressException
from volatility3.framework.layers.intel import Intel32e
from volatility3.framework.interfaces.layers import ScannerInterface
from volatility3.cli import PrintedProgress

COLUMNS = [
    (
        "This plugin writes to a csv file instead of stdout because volatilites csv renderer is excruciatingly slow...",
        str,
    )
]

# DEFAULT_DTB_ADDR = 0x888000000000
DEFAULT_DTB_ADDR = 0x1e0a000

# Canonical 64 bit "high mem" pointers
# Using the infamous lookahead and zero-length capture to get overlapping matches.
# https://stackoverflow.com/questions/5616822/python-regex-find-all-overlapping-matches
HIGHMEM_POINTER_PATTERN = re.compile(b"(?=(.{5}[\x80-\xFF]\xFF\xFF))", flags=re.DOTALL)
POINTER_SIZE = 8
CSV_SEP = ","
CSV_COLUMNS = ["offset", "virtual", "physical"]


class HighMemPointerScanner(ScannerInterface):
    """
    Looks for "high mem" pointers. (8 byte aligned little endian ints with bit 63 to 48 set to 1.)
    """

    thread_safe = True

    def __call__(self, data: bytes, data_offset: int) -> Generator[int, None, None]:
        for match in HIGHMEM_POINTER_PATTERN.finditer(data):
            offset = match.start(1)
            if offset < self.chunk_size and offset % POINTER_SIZE == 0:
                yield offset + data_offset


class HighmemPointerScan(plugins.PluginInterface):
    _required_framework_version = (1, 0, 0)

    @classmethod
    def get_requirements(cls) -> List[interfaces.configuration.RequirementInterface]:
        return [
            requirements.TranslationLayerRequirement(
                name="primary",
                description="Memory layer for the kernel",
            ),
            requirements.IntRequirement(
                name="dtb",
                optional=True,
                default=DEFAULT_DTB_ADDR,
                description="Dtb physical offset",
            ),
            requirements.StringRequirement(
                name="outfile",
                optional=False,
                description="Path to store results at. (CSV format) The results are not written to stdout because "
                "Volatilities csv renderer is excruciatingly slow.",
            ),
        ]

    def add_layer(self, memory_layer: str):
        new_layer_name = self.context.layers.free_layer_name("ManualIntelLayer")
        config_path = join("PointerScan", new_layer_name)
        self.context.config[join(config_path, "memory_layer")] = memory_layer
        self.context.config[join(config_path, "page_map_offset")] = self.config["dtb"]
        layer = Intel32e(self.context, config_path=config_path, name=new_layer_name)
        self.context.add_layer(layer)
        return layer

    @functools.lru_cache(maxsize=None)
    def dir2base(self, layer, dir, index):
        entry = struct.unpack("<Q", layer.read(dir + 8*index, 8))[0]
        if(entry&1 ==0):  # not present
            raise InvalidAddressException("dir2base", dir, "Page not present")
        return (entry&0x001FFFFFFFFFF000, entry&0xFFF)

    @functools.lru_cache(maxsize=None)
    def translate(self, layer, dtb, vaddr):
        (l4,f4) = self.dir2base(layer, dtb,  (vaddr>>39)&0x1FF);
        (l3,f3) = self.dir2base(layer, l4, (vaddr>>30)&0x1FF);
        if(f3&0x80):
            return l3 + (vaddr&((1<<30)-1))
        (l2,f2) = self.dir2base(layer, l3, (vaddr>>21)&0x1FF);
        if(f2&0x80):
            return l2 + (vaddr&((1<<21)-1))
        (l1,f1) = self.dir2base(layer, l2, (vaddr>>12)&0x1FF);
        paddr = l1 + (vaddr&((1<<12)-1))
        # to check if paddr is in layer
        # if commented out, will also return physical addresses in memory mapping that are outside of memory dump (iomem??)
        # layer.read(paddr, 8)
        return paddr

    def _generator(self):
        layer: Intel32e = self.context.layers[self.config["primary"]]

        # Choose the highest non-virtual layer:
        # https://github.com/volatilityfoundation/volatility3/issues/486
        while isinstance(layer, layers.intel.Intel):
            layer = self.context.layers[layer.config["memory_layer"]]

        translation_layer = self.add_layer(memory_layer=layer.name)
        dtb = self.config["dtb"]

        results = []

        for offset in layer.scan(
            context=self.context,
            scanner=HighMemPointerScanner(),
            progress_callback=PrintedProgress(),
        ):
            address = struct.unpack("<Q", layer.read(offset, 8))[0]
            # You need to "mask" addresses before passing them to translate because volatility
            # discards the first 16 bits of vaddresses. See volatility.framework.layers.intel.Intel32e.translate.
            address_masked = address & translation_layer.address_mask
            try:
                # paddr = translation_layer.translate(address_masked)[0]
                paddr = self.translate(layer, dtb, address_masked)
                results.append((hex(offset), hex(address), hex(paddr)))
            except InvalidAddressException:
                results.append((hex(offset), hex(address), ""))

        with open(self.config["outfile"], "w+", newline="") as f:
            writer = csv.writer(f, dialect="unix", quoting=csv.QUOTE_NONE)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(results)
        yield 0, (f"Done. Pointers written to {self.config['outfile']}",)

    def run(self):
        return renderers.TreeGrid(COLUMNS, self._generator())
