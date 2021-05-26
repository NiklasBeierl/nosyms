import pandas as p
import pickle
from time import time
from encoding.memory_graph import Pointer, build_memory_graph
from encoding import SandwichEncoder

POINTER_SIZE = 8
df = p.read_csv("./data_dump/memory_layer_nokaslr_pointers_translated.csv")
df = df.dropna()
pointers = [Pointer(*tup) for tup in df[["offset", "physical"]].itertuples(index=False)]

start = time()
het_g = build_memory_graph(pointers, SandwichEncoder("./data_dump/memory_layer_nokaslr.raw", POINTER_SIZE))
print(f"Encoding took {time() - start} seconds")
with open("./memory_graph.pkl", "wb+") as f:
    pickle.dump(het_g, f)

print("Done")
