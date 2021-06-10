import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import torch as t
from torch.nn.functional import cross_entropy, one_hot
from sklearn.model_selection import train_test_split
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from networks.embedding import MyConvolution
from networks.utils import one_hot_with_neutral, add_self_loops
from encoding import BlockType
from file_paths import SYM_DATA_PATH, MODEL_PATH
from hyperparams import *
import develop.filter_warnings

TARGET_SYMBOL = "task_struct"
BINARY_CLASSIFY = True

all_graphs = []
all_syms = list(Path(SYM_DATA_PATH).glob("*.pkl"))
all_syms = all_syms[::2]  # Need more RAM!
print(f"Using: {all_syms}")
for path in all_syms:
    with open(path, "rb") as f:
        graph, node_ids = pickle.load(f)
    true_labels = np.zeros(graph.num_nodes(), dtype=np.int8)
    for i, n in node_ids.inv.items():
        if TARGET_SYMBOL in n.type_descriptor:  # TODO This is not he safest check, but it works for task_struct :)
            true_labels[i] = 1 if BINARY_CLASSIFY else i
    graph.ndata[f"{TARGET_SYMBOL}_labels"] = t.tensor(true_labels)
    all_graphs.append(graph)

batch_graph = dgl.batch(all_graphs)
del all_graphs  # Free Memory
batch_graph = add_self_loops(batch_graph, etype=SELF_LOOPS)
labels = batch_graph.ndata[f"{TARGET_SYMBOL}_labels"]

in_size = batch_graph.ndata["blocks"].shape[1] * (len(BlockType) - 1)  # Unknowns are either randomized or neutered
out_size = int(max(labels) + 1)
hidden_size = int(np.median([in_size, out_size]))

model_etypes = ["follows", "precedes", "pointed_to_by"]  # Ignoring the "is" relationship
model = MyConvolution(model_etypes, in_size, hidden_size, out_size)
# https://github.com/dmlc/dgl/issues/2310
# https://github.com/dmlc/dgl/issues/3003
batch_graph.remove_edges(batch_graph.edge_ids(*batch_graph.edges(etype="is"), etype="is"), etype="is")
batch_graph.remove_edges(batch_graph.edge_ids(*batch_graph.edges(etype="is"), etype="is"), etype="is")

index = np.array(range(batch_graph.num_nodes()))
train_idx, test_idx = train_test_split(index, random_state=33, train_size=0.7, stratify=labels)
batch_graph.ndata["train"] = t.zeros(batch_graph.num_nodes()).bool()
batch_graph.ndata["train"][train_idx] = True

opt = t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
best_test_acc = 0

loss_weights = t.full((int(labels.max()) + 1,), 1, dtype=t.float)
label_count = Counter(labels.numpy())
loss_weights[1] = label_count[0] / label_count[1]
dev = t.device("cpu")

if t.cuda.is_available():
    print("Going Cuda!")
    dev = t.device("cuda:0")
    labels = batch_graph.ndata[f"{TARGET_SYMBOL}_labels"]
    model.cuda(dev)
    loss_weights = loss_weights.cuda(dev)

if UNKNOWN == "randomize":
    gen = t.Generator().manual_seed(90324590320)
    unknown_idx = batch_graph.ndata["blocks"] == 0
    # Subtracting one since the 0 class will be "randomized away"
    original_blocks = batch_graph.ndata["blocks"].clone() - 1

elif UNKNOWN == "neutral":
    blocks_one_hot = one_hot_with_neutral(batch_graph.ndata["blocks"].long())
    blocks_one_hot = blocks_one_hot.reshape(blocks_one_hot.shape[0], -1)
    batch_graph.ndata["blocks"] = blocks_one_hot.float()


for epoch in range(EPOCHS):

    # Randomize unknown elements
    if UNKNOWN == "randomize":
        random_blocks = t.randint(0, 3, original_blocks.shape, generator=gen, dtype=t.int8)
        original_blocks[unknown_idx] = random_blocks[unknown_idx]
        blocks_one_hot = one_hot(original_blocks.long()).reshape(original_blocks.shape[0], -1)
        batch_graph.ndata["blocks"] = blocks_one_hot.float()

    epoch_results = t.full((batch_graph.num_nodes(), out_size), float("inf"), dtype=t.float32)
    sampler = MultiLayerFullNeighborSampler(BALL_CONV_LAYERS)
    loader = NodeDataLoader(batch_graph, np.arange(batch_graph.num_nodes()), sampler, batch_size=BATCH_SIZE)

    for i, chunk in enumerate(loader):
        input_nodes, output_nodes, blocks = chunk
        blocks = [b.to(dev) for b in blocks]
        batch_labels = blocks[-1].dstdata[f"{TARGET_SYMBOL}_labels"]
        batch_train_idx = blocks[-1].dstdata["train"]
        batch_logits = model.forward_batch(blocks)
        batch_loss = cross_entropy(batch_logits[batch_train_idx], batch_labels[batch_train_idx], weight=loss_weights)
        batch_pred = batch_logits.argmax(1)
        epoch_results[output_nodes, :] = batch_logits.detach().cpu()

        opt.zero_grad()
        batch_loss.backward()
        opt.step()

    loss = cross_entropy(epoch_results, labels, weight=loss_weights.cpu())
    pred = epoch_results.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    if epoch % 5 == 0:
        print(
            f"{epoch} Loss {loss.item():.4f}, Train Acc {train_acc.item():.4f}, Test Acc {test_acc.item():.4f}"
            + f"(Best {best_test_acc.item():.4f})"
        )

with open(MODEL_PATH, "wb+") as f:
    pickle.dump(model, f)

print("Done.")
