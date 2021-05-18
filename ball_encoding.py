import json
import pickle
from encoding import VolatilitySymbolsEncoder
from encoding.ball import BallGraphBuilder

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")

target_user_type = "task_struct"
path = "./data_dump/vmlinux-5.4.0-58-generic.json"

with open(path) as f:
    syms = json.load(f)
sym_encoder = VolatilitySymbolsEncoder(syms)

ball_encoder = BallGraphBuilder()

graph, node_ids = ball_encoder.create_type_graph(sym_encoder, target_user_type)

with open("./ball-sym-data.pkl", "wb+") as f:
    pickle.dump((graph, node_ids), f)

print("Done")
