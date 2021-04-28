import pandas as p
from encoding.snapshot import Pointer, build_dgl_graph
from encoding.memory import SandwichEncoder

df = p.read_csv("./data_dump/memory_layer_nokaslr_pointers_translated.csv")
df = df.dropna()
pointers = [Pointer(*tup) for tup in df[["offset", "physical"]].itertuples(index=False)]

het_g = build_dgl_graph(pointers, SandwichEncoder("./data_dump/memory_layer_nokaslr.raw"))

print("Done")