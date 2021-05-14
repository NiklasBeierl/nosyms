import struct
import csv
from typing import List
from volatility.framework import renderers, interfaces, layers
from volatility.framework.configuration import requirements
from volatility.framework.interfaces import plugins
from volatility.framework.interfaces.configuration import path_join as join
from volatility.framework.exceptions import InvalidAddressException
from volatility.framework.layers.intel import Intel32e
from volatility.framework.layers.scanners import RegExScanner
from volatility.cli import PrintedProgress

COLUMNS = [
    (
        "This plugin writes to a csv file instead of stdout because volatilites csv renderer is excruciatingly slow...",
        str,
    )
]

DEFAULT_DTB_ADDR = 0x888000000000

# Canonical 64 bit "high mem" pointers
POINTER_PATTERN = b".{5}[\x80-\xFF]\xFF\xFF"
CSV_SEP = ","
CSV_COLUMNS = ["offset", "virtual", "physical"]


class HighmemPointerScan(plugins.PluginInterface):
    _required_framework_version = (2, 0, 0)

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

    def _generator(self):
        layer: Intel32e = self.context.layers[self.config["primary"]]

        # Choose the highest non-virtual layer:
        # https://github.com/volatilityfoundation/volatility3/issues/486
        while isinstance(layer, layers.intel.Intel):
            layer = self.context.layers[layer.config["memory_layer"]]

        translation_layer = self.add_layer(memory_layer=layer.name)

        results = []
        for offset in layer.scan(
            context=self.context,
            scanner=RegExScanner(POINTER_PATTERN),
            progress_callback=PrintedProgress(),
        ):
            address = struct.unpack("<Q", layer.read(offset, 8))[0]
            # You need to "mask" addresses before passing them to translate because volatility
            # discards the first 16 bits of vaddresses. See volatility.framework.layers.intel.Intel32e.translate.
            address_masked = address & translation_layer.address_mask
            try:
                paddr = translation_layer.translate(address_masked)[0]
                results.append((offset, address, paddr))
            except InvalidAddressException:
                results.append((offset, address, ""))

        with open(self.config["outfile"], "w+", newline="") as f:
            writer = csv.writer(f, dialect="unix", quoting=csv.QUOTE_NONE)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(results)
        yield 0, (f"Done. Pointers written to {self.config['outfile']}",)

    def run(self):
        return renderers.TreeGrid(COLUMNS, self._generator())
