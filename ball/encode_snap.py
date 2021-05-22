import pandas as pd
from encoding import Pointer
from encoding.ball import BallEncoder, BallGraphBuilder

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")

pointers_df = pd.read_csv("../data_dump/nokaslr_pointers.csv").dropna()

pointers = [Pointer(o, t) for o, t in pointers_df[["offset", "physical"]].itertuples(index=False)]
encoder = BallEncoder("../data_dump/nokaslr.raw", pointer_size=8)
bgb = BallGraphBuilder()

memgraph = bgb.create_snapshot_graph(encoder, pointers)

import pickle

with open("./ball-mem-graph.pkl", "wb+") as f:
    pickle.dump(memgraph, f)

print("Done.")
