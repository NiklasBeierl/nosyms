import pandas as pd
from encoding import Pointer
from encoding.ball import BallEncoder, BallGraphBuilder
from encoding import WordCompressor, VolatilitySymbolsEncoder

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")

pointers_df = pd.read_csv("../data_dump/nokaslr_pointers.csv").dropna()
pointers = [Pointer(o, t) for o, t in pointers_df[["offset", "physical"]].itertuples(index=False)]

encoder = BallEncoder("../data_dump/nokaslr.raw", pointers=[p.offset for p in pointers], pointer_size=8)
bgb = BallGraphBuilder()
comp = WordCompressor()

mem_graph = bgb.create_snapshot_graph(encoder, pointers, comp)

memgraph = bgb.create_snapshot_graph(encoder, pointers)

import pickle

with open("./ball-mem-graph.pkl", "wb+") as f:
    pickle.dump(mem_graph, f)

print("Done.")
