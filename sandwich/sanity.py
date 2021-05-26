import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from encoding.sandwich import SandwichEncoder
from encoding import VolatilitySymbolsEncoder, BlockType

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

POINTER_SIZE = 8
with open("./data_dump/vmlinux-5.4.0-58-generic.json") as f:
    sym_encoder = VolatilitySymbolsEncoder(json.load(f))

tasks = pd.read_csv("./data_dump/memory_layer_nokaslr_tasks.csv")
mem_encoder = SandwichEncoder("./data_dump/memory_layer_nokaslr.raw", POINTER_SIZE)
ts_sym_blocks, _ = sym_encoder.encode_user_type("task_struct")
ts_def = sym_encoder.syms["user_types"]["task_struct"]
ts_size = ts_def["size"]
ts_list_head_offset = ts_def["fields"]["tasks"]["offset"]

conflicts = defaultdict(lambda: 0)
correct = defaultdict(lambda: 0)
eq_qs = []
for i in range(len(tasks)):
    task_addr = tasks["next_p"][i] - ts_list_head_offset
    task = mem_encoder.encode(task_addr - 8, task_addr + ts_size + 9)
    task = task[9:-8]

    corr = 0
    for a, b in zip(ts_sym_blocks, task):
        if a != b and a != BlockType.Pointer:  # We can't get pointers right with the sandwich encoder.
            conflicts[(a, b)] += 1
        else:
            corr += 1
            correct[a] += 1

    eq_q = corr / len(ts_sym_blocks)
    eq_qs.append(eq_q)

print(f"Mean eqq: {np.mean(eq_qs)}")
print(f"Std eqq: {np.std(eq_qs)}")


with open("./memory_graph.pkl", "rb") as f:
    mem_g = pickle.load(f)

ts_excerpt = ts_sym_blocks[ts_list_head_offset : ts_list_head_offset + (16 * 8)]
raw = mem_encoder.mmap[tasks.next_p[0] : tasks.next_p[0] + (16 * 8)]

chunk = (mem_g.ndata["start"] == tasks.next_p[0]).nonzero(as_tuple=True)[0]
mem_g.ndata["blocks"][chunk + 6].reshape(100, 3)[:24]


print("Done")
