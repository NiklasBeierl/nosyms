import pickle
import warnings
from torch.nn.functional import softmax

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")


with open("./ball-mem-graph.pkl", "rb") as f:
    mem_graph = pickle.load(f)


with open("./results.pkl", "rb") as f:
    results = pickle.load(f)
    if results.is_cuda:
        results = results.cpu()
    results = softmax(results)
    results = results.detach().numpy()

yay = results[mem_graph.ndata["pids"] != -1, 1].mean()
nay = results[mem_graph.ndata["pids"] == -1, 0].mean()

tasks_results = results[mem_graph.ndata["pids"] != -1]

# This file only really makes sense if used in a debugger with facilities to properly visualise tensors / arrays.
print("Done")
