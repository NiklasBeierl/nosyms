import json
import pandas as pd
from encoding import SymbolEncoder
from volatility.framework import contexts
from volatility.framework.symbols import linux

sym_path = "//home/niklas/code/nosyms/symbols/all_syms/2.6.15-1.2054_FC5.x86_64.json"
ctx = contexts.Context()
table_name = ctx.symbol_space.free_table_name("BlockTypeEncoder")
table = linux.LinuxKernelIntermedSymbols(ctx, table_name, name=table_name, isf_url="file://" + sym_path)
# ctx.symbol_space.append(table) # TODO: Necessary?

task_struct_sym = table.get_type("task_struct")

with open(sym_path) as f:
    syms_json = json.load(f)

task_struct_json = syms_json["user_types"]["task_struct"]
task_struct_fields = pd.DataFrame(task_struct_json["fields"]).T.sort_values("offset")

encoder = SymbolEncoder(syms_json)
atomic_task_struct = encoder.user_type_to_atomic("task_struct")

print("Done")
