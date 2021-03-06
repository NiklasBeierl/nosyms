import pickle
import torch as t
import dgl.dataloading as dgldl
from torch.nn.functional import one_hot
from hyperparams import BALL_CONV_LAYERS
from nosyms.nn.utils import add_self_loops
from file_paths import MODEL_PATH, MEM_GRAPH_PATH, RESULTS_PATH
from nosyms.encoding import BlockType
import develop.filter_warnings

BATCH_SIZE = 1000

with open(MEM_GRAPH_PATH, "rb") as f:
    mem_graph = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    model.training = False

mem_graph = add_self_loops(mem_graph)

blocks = mem_graph.ndata["blocks"]
unknown_idx = blocks == BlockType.Unknown.value
blocks -= 1
blocks[unknown_idx] = t.randint(0, 3, blocks.shape, dtype=t.int8)[unknown_idx]  # There are going to be few if any.
blocks_one_hot = one_hot(blocks.long())
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
    results[output_nodes.long(), :] = batch_result.detach().cpu()
    perc = BATCH_SIZE * i * 100 // mem_graph.num_nodes()
    if perc % 5 == 0 and perc != last_prec_done:
        print(f"{perc}% done.")
        last_prec_done = perc


with open(RESULTS_PATH, "wb+") as f:
    pickle.dump(results, f)

print("Done")
