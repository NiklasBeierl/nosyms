import pickle
import json
import numpy as np
import pandas as pd
import torch as t
from interlap import InterLap
from nosyms.encoding import Pointer
from nosyms.encoding.ball import BallEncoder, BallGraphBuilder
from nosyms.encoding import WordCompressor
from file_paths import MATCHING_SYMBOLS_PATH, MEM_GRAPH_PATH, TASKS_CSV_PATH, POINTER_CSV_PATH, RAW_DUMP_PATH
from warnings import warn
import develop.filter_warnings

pointers_df = pd.read_csv(POINTER_CSV_PATH).dropna()
pointers_df.physical = pointers_df.physical.astype(int)
pointers = [Pointer(o, t) for o, t in pointers_df[["offset", "physical"]].itertuples(index=False)]

encoder = BallEncoder(RAW_DUMP_PATH, pointers=[p.offset for p in pointers], pointer_size=8)
bgb = BallGraphBuilder()
comp = WordCompressor()

mem_graph = bgb.create_snapshot_graph(encoder, pointers, comp)

with open(MATCHING_SYMBOLS_PATH) as f:
    syms = json.load(f)
ts_def = syms["user_types"]["task_struct"]
ts_size = ts_def["size"]

tasks = pd.read_csv(TASKS_CSV_PATH)
task_index = np.full(mem_graph.num_nodes(), -1, dtype=np.int32)

inter = InterLap()
chunks = list(zip(mem_graph.ndata["start"].numpy(), mem_graph.ndata["end"].numpy(), range(mem_graph.num_nodes())))
inter.update(chunks)
for pid, offset in tasks[["PID", "physical"]].itertuples(index=False):
    for cs, ce, i in inter.find((offset, offset + ts_size)):
        if task_index[i] != -1:
            warn("A chunk is covering multiple task structs. 'pids' labels in mem graph will be inaccurate")
        task_index[i] = pid

mem_graph.ndata["pids"] = t.tensor(task_index)

with open(MEM_GRAPH_PATH, "wb+") as f:
    pickle.dump(mem_graph, f)

print("Done.")
