import pickle
import json
from collections import Counter
from pathlib import Path
import datetime as dt
import numpy as np
import torch as t
from torch.nn.functional import cross_entropy, one_hot
from sklearn.model_selection import train_test_split
import dgl
from dgl.sampling import sample_neighbors
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from nosyms.nn.models import MyConvolution
from nosyms.nn.utils import add_self_loops
from nosyms.encoding import BlockType
from file_paths import SYM_DATA_PATH, MODEL_PATH
from hyperparams import *
import develop.filter_warnings

TARGET_SYMBOL = "task_struct"
BINARY_CLASSIFY = True

FANOUT = {
    "pointed_to_by": -1,  # If a node exists, it will point somewhere.
    "precedes": 1,  # In a mem graph, there is one precedes at max
    "follows": 1,  # In a mem graph, there is one follows at max
    "is": -1,  # Removed all edges anyways
}


all_graphs = []
all_syms = list(Path(SYM_DATA_PATH).glob("vmlinux*.pkl"))

labels = np.array([])
in_size = None


print(f"Training script start time: {dt.datetime.now()}")

print(f"Using: {all_syms}")
for path in all_syms:
    with open(path, "rb") as f:
        graph, node_ids = pickle.load(f)

    local_in_size = graph.ndata["blocks"].shape[1] * (len(BlockType) - 1)  # Unknowns are either randomized or neutered

    if not in_size:
        in_size = local_in_size
    elif in_size != local_in_size:
        raise ValueError("Trying to combine graphs of different block type vector lenght.")

    local_labels = np.zeros(graph.num_nodes(), dtype=np.int8)
    for i, n in node_ids.inv.items():
        if json.dumps({"kind": "struct", "name": TARGET_SYMBOL}) == n.type_descriptor:
            local_labels[i] = 1 if BINARY_CLASSIFY else i

    offset = len(labels)
    all_graphs.append((path, offset, offset + graph.num_nodes()))
    labels = np.concatenate([labels, local_labels])


num_nodes = len(labels)
out_size = int(max(labels) + 1)
hidden_size = int(np.median([in_size, out_size]))

model_etypes = ["follows", "precedes", "pointed_to_by"]  # Ignoring the "is" relationship
model = MyConvolution(model_etypes, BALL_CONV_LAYERS, in_size, hidden_size, out_size)

index = np.array(range(num_nodes))
train_idx, test_idx = train_test_split(index, random_state=33, train_size=0.7, stratify=labels)
full_train_idx = np.zeros(num_nodes, dtype=bool)
full_train_idx[train_idx] = True

opt = t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
best_test_acc = 0

loss_weights = t.full((int(max(labels)) + 1,), 1, dtype=t.float)
label_count = Counter(labels)
loss_weights[1] = label_count[0] / label_count[1]
dev = t.device("cpu")

gen = t.Generator().manual_seed(90324590320)
labels = t.tensor(labels)

if t.cuda.is_available():
    print("Going Cuda!")
    dev = t.device("cuda:0")
    model.cuda(dev)
    loss_weights = loss_weights.cuda(dev)


def prepare_graph(path, start_idx, end_idx):
    with open(path, "rb") as f:
        graph, _ = pickle.load(f)
    graph = add_self_loops(graph, etype=SELF_LOOPS)

    # https://github.com/dmlc/dgl/issues/2310
    # https://github.com/dmlc/dgl/issues/3003
    graph.remove_edges(graph.edge_ids(*graph.edges(etype="is"), etype="is"), etype="is")
    graph.remove_edges(graph.edge_ids(*graph.edges(etype="is"), etype="is"), etype="is")

    graph.ndata["train"] = t.tensor(full_train_idx[start_idx:end_idx])

    graph.ndata[f"{TARGET_SYMBOL}_labels"] = labels[start_idx:end_idx]

    unknown_idx = graph.ndata["blocks"] == BlockType.Unknown.value
    blocks = graph.ndata["blocks"] - 1
    blocks[unknown_idx] = t.randint(0, 3, blocks.shape, generator=gen, dtype=t.int8)[unknown_idx]
    blocks = one_hot(blocks.long()).reshape(blocks.shape[0], -1)
    graph.ndata["blocks"] = blocks.float()
    return graph


print(f"Training start time: {dt.datetime.now()}")
for epoch in range(EPOCHS):

    epoch_results = t.full((num_nodes, out_size), float("inf"), dtype=t.float32)

    for path, start_idx, end_idx in all_graphs:
        graph = prepare_graph(path, start_idx, end_idx)
        graph = sample_neighbors(graph, np.arange(graph.num_nodes()), FANOUT)

        sampler = MultiLayerFullNeighborSampler(BALL_CONV_LAYERS)
        loader = NodeDataLoader(graph, np.arange(graph.num_nodes()), sampler, batch_size=BATCH_SIZE)

        for input_nodes, output_nodes, blocks in loader:
            blocks = [b.to(dev) for b in blocks]
            batch_labels = blocks[-1].dstdata[f"{TARGET_SYMBOL}_labels"].long()
            batch_train_idx = blocks[-1].dstdata["train"]
            batch_logits = model.forward_batch(blocks)
            batch_loss = cross_entropy(
                batch_logits[batch_train_idx], batch_labels[batch_train_idx], weight=loss_weights
            )
            batch_pred = batch_logits.argmax(1)
            output_nodes = output_nodes.clone() + start_idx
            epoch_results[output_nodes.long(), :] = batch_logits.detach().cpu()

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

    loss = cross_entropy(epoch_results, labels.long(), weight=loss_weights.cpu())
    pred = epoch_results.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    if epoch % 5 == 0:
        print(
            f"{epoch} Loss {loss.item():.4f}, Train Acc {train_acc.item():.4f}, Test Acc {test_acc.item():.4f}"
            + f"(Best {best_test_acc.item():.4f}) - {dt.datetime.now()}"
        )

with open(MODEL_PATH, "wb+") as f:
    pickle.dump(model, f)

print("Done.")
