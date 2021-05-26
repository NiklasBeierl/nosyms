import pickle
import json
import warnings
import pandas as pd
import numpy as np
from rbi_tree.tree import ITree
from encoding import VolatilitySymbolsEncoder

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


with open("./ball-mem-graph.pkl", "rb") as f:
    mem_graph = pickle.load(f)


with open("./results.pkl", "rb") as f:
    results = pickle.load(f).detach().numpy()

tasks = pd.read_csv("../data_dump/nokaslr_tasks.csv")
# pointers = pd.read_csv("./data_dump/memory_layer_nokaslr_pointers_translated.csv")

with open("../data_dump/vmlinux-5.4.0-58-generic.json") as f:
    sym_encoder = VolatilitySymbolsEncoder(json.load(f))
ts_def = sym_encoder.syms["user_types"]["task_struct"]
ts_size = ts_def["size"]
fields_ordered = list(sorted(ts_def["fields"], key=lambda field: field["offset"]))


itr = ITree()

for a, b, i in zip(mem_graph.ndata["start"], mem_graph.ndata["end"], range(mem_graph.num_nodes())):
    itr.insert(a, b, i)

task_offsets = list(sorted(tasks.physical))
task_index = np.zeros(mem_graph.num_nodes(), dtype=int)
for tid, t_o in enumerate(task_offsets):
    for cs, ce, i in itr.find(t_o, t_o + ts_size):
        task_index[i] = tid


frist_task = results[task_index == 1]

# This file only really makes sense if used in a debugger with facilities to properly visualise tensors / arrays.
print("Done")
