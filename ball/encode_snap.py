import pickle
import json
import numpy as np
import pandas as pd
import torch as t
from rbi_tree.tree import ITree
from encoding import Pointer
from encoding.ball import BallEncoder, BallGraphBuilder
from encoding import WordCompressor, VolatilitySymbolsEncoder
from file_paths import MATCHING_SYMBOLS_PATH, MEM_GRAPH_PATH, TASKS_CSV_PATH, POINTER_CSV_PATH, RAW_DUMP_PATH

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")

pointers_df = pd.read_csv(POINTER_CSV_PATH).dropna()
pointers = [Pointer(o, t) for o, t in pointers_df[["offset", "physical"]].itertuples(index=False)]

encoder = BallEncoder(RAW_DUMP_PATH, pointers=[p.offset for p in pointers], pointer_size=8)
bgb = BallGraphBuilder()
comp = WordCompressor()

mem_graph = bgb.create_snapshot_graph(encoder, pointers, comp)

chunk_tree = ITree()
for a, b, i in zip(mem_graph.ndata["start"], mem_graph.ndata["end"], range(mem_graph.num_nodes())):
    chunk_tree.insert(a, b, i)

with open(MATCHING_SYMBOLS_PATH) as f:
    sym_encoder = VolatilitySymbolsEncoder(json.load(f))
ts_def = sym_encoder.syms["user_types"]["task_struct"]
ts_size = ts_def["size"]

tasks = pd.read_csv(TASKS_CSV_PATH)
task_index = np.full(mem_graph.num_nodes(), -1, dtype=np.int32)
for pid, offset in tasks[["PID", "physical"]].itertuples(index=False):
    for cs, ce, i in chunk_tree.find(offset, offset + ts_size):
        # TODO: A chunk MIGHT cover two task_structs!
        task_index[i] = pid

mem_graph.ndata["pids"] = t.tensor(task_index)

with open(MEM_GRAPH_PATH, "wb+") as f:
    pickle.dump(mem_graph, f)

print("Done.")
