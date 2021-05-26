import pickle
from collections import deque
import json
import pandas as pd
import torch as t
import warnings
from encoding import VolatilitySymbolsEncoder

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

with open("./memory_graph.pkl", "rb") as f:
    mem_graph = pickle.load(f)


with open("./classify_results.pkl", "rb") as f:
    results = t.load(f)

tasks = pd.read_csv("./data_dump/memory_layer_nokaslr_tasks.csv")
pointers = pd.read_csv("./data_dump/memory_layer_nokaslr_pointers_translated.csv")

with open("./data_dump/vmlinux-5.4.0-58-generic.json") as f:
    sym_encoder = VolatilitySymbolsEncoder(json.load(f))
ts_def = sym_encoder.syms["user_types"]["task_struct"]
ts_list_head_offset = ts_def["fields"]["tasks"]["offset"]


task_index = []
current_slice = 0
task_offsets = list(sorted(lh - ts_list_head_offset for lh in tasks.next_p))
slices = list(zip(mem_graph.ndata["start"], mem_graph.ndata["end"]))

for to in task_offsets:
    while to > slices[current_slice][0]:
        current_slice += 1
        task_index.append(False)
    task_index.append(True)
    current_slice += 1

task_index += [False] * (len(results) - len(task_index))

ti_rotatable = deque(task_index)
eval_mat = []
for _ in range(results.shape[1]):
    eval_mat.append(t.mean(results[list(ti_rotatable)], dim=0))
    ti_rotatable.rotate()

eval_mat = t.stack(eval_mat)

eval_mat_df = pd.DataFrame(eval_mat.detach().numpy())


print("Done")
