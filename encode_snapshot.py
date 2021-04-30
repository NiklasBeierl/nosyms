import pandas as p
from encoding.memory_graph import Pointer, build_memory_graph
from encoding import SandwichEncoder

df = p.read_csv("./data_dump/memory_layer_nokaslr_pointers_translated.csv")
df = df.dropna()
pointers = [Pointer(*tup) for tup in df[["offset", "physical"]].itertuples(index=False)]

het_g = build_memory_graph(pointers, SandwichEncoder("./data_dump/memory_layer_nokaslr.raw"))

print("Done")
