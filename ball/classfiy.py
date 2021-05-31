import pickle
import torch as t
import numpy as np
import dgl.dataloading as dgldl
from torch.nn.functional import softmax
from hyperparams import BALL_CONV_LAYERS
from networks.utils import one_hot_with_neutral
import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

BATCH_SIZE = 3000

with open("./ball-mem-graph.pkl", "rb") as f:
    mem_graph = pickle.load(f)

with open("./model.pkl", "rb") as f:
    model = pickle.load(f)

blocks_one_hot = one_hot_with_neutral(mem_graph.ndata["blocks"].long())
blocks_one_hot = blocks_one_hot.reshape(blocks_one_hot.shape[0], -1)
mem_graph.ndata["blocks"] = blocks_one_hot.float()

output_size = model.class_layers[-1].out_features
results = t.full((mem_graph.num_nodes(), output_size), float("inf"), dtype=t.float32)
sampler = dgldl.MultiLayerFullNeighborSampler(BALL_CONV_LAYERS)
loader = dgldl.NodeDataLoader(
    mem_graph,
    list(range(mem_graph.num_nodes())),
    sampler,
    batch_size=BATCH_SIZE,
)

last_prec_done = 0
for i, chunk in enumerate(loader):
    input_nodes, output_nodes, blocks = chunk
    blocks = [b.to(t.device("cuda")) for b in blocks]
    batch_result = model.forward_batch(blocks)
    results[output_nodes, :] = batch_result.detach().cpu()
    perc = BATCH_SIZE * i * 100 // mem_graph.num_nodes()
    if perc % 5 == 0 and perc != last_prec_done:
        print(f"{perc}% done.")
        last_prec_done = perc


with open("./results.pkl", "wb+") as f:
    pickle.dump(results, f)

print("Done")
