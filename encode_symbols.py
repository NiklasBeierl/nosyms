import json
import pandas as pd
from encoding import VolatilitySymbolsEncoder
from encoding.symbol_graph import build_vol_symbols_graph

sym_path = "./symbols/all_syms/2.6.15-1.2054_FC5.x86_64.json"

with open(sym_path) as f:
    syms_json = json.load(f)

task_struct_json = syms_json["user_types"]["task_struct"]
task_struct_fields = pd.DataFrame(task_struct_json["fields"]).T.sort_values("offset")

encoder = VolatilitySymbolsEncoder(syms_json)
graph = build_vol_symbols_graph("task_struct", encoder)
blocks, pointers = encoder.encode_user_type("task_struct")

print("Done")
