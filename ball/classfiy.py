import pickle
from torch.nn.functional import one_hot
import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

with open("./ball-mem-graph.pkl", "rb") as f:
    mem_graph = pickle.load(f)

with open("./model.pkl", "rb") as f:
    model = pickle.load(f)

print("Loaded stuff.")

blocks_one_hot = one_hot(mem_graph.ndata["blocks"].long())
blocks_one_hot = blocks_one_hot.reshape(blocks_one_hot.shape[0], -1)
mem_graph.ndata["blocks"] = blocks_one_hot.float()

results = model(mem_graph)

with open("./results.pkl", "wb+") as f:
    pickle.dump(results, f)

print("Done")
