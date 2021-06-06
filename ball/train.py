import pickle
from collections import Counter
import warnings
import numpy as np
import torch as t
from torch.nn.functional import cross_entropy
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
import dgl
from networks.embedding import MyConvolution
from networks.utils import one_hot_with_neutral, add_self_loops
from encoding import WordCompressor
from file_paths import SYM_DATA_PATH, MODEL_PATH
from hyperparams import EPOCHS, LEARNING_RATE

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")

with open(SYM_DATA_PATH, "rb") as f:
    all_data = pickle.load(f)

TARGET_SYMBOL = "task_struct"
BINARY_CLASSIFY = True


def encoding_rand_score(graph, true_labels):
    encodings = {}
    encoding_labels = np.zeros(graph.num_nodes())
    for i, bs in enumerate(graph.ndata["blocks"]):
        key = tuple(bs.numpy())
        if key not in encodings:
            encodings[key] = len(encodings)
        encoding_labels[i] = encodings[key]
    # Many balls end up with identical type encodings (not necessarily identical edges, tho)
    return rand_score(true_labels, encoding_labels)


compressor = WordCompressor()
rands_comp = []
rands = []
for path, graph, node_ids in all_data:
    true_labels = np.zeros(graph.num_nodes())
    for i, n in node_ids.inv.items():
        if TARGET_SYMBOL in n.type_descriptor:  # TODO This is not he safest check, but it works for task_struct :)
            true_labels[i] = 1 if BINARY_CLASSIFY else i

    graph.ndata[f"{TARGET_SYMBOL}_labels"] = t.tensor(true_labels)
    r_score = encoding_rand_score(graph, true_labels)
    rands.append(r_score)
    graph.ndata["blocks"] = compressor.compress_batch(graph.ndata["blocks"])
    r_score_compressed = encoding_rand_score(graph, true_labels)
    rands_comp.append(r_score_compressed)

print(
    f"""Rand score for encodings:
      Original:   Compressed:   
mean: {np.mean(rands):.4f}      {np.mean(rands_comp):.4f} 
std:  {np.std(rands):.4f}      {np.std(rands_comp):.4f}"""
)

batch_graph = dgl.batch([g for _, g, _ in all_data])
batch_graph = add_self_loops(batch_graph)
blocks_one_hot = one_hot_with_neutral(batch_graph.ndata["blocks"].long())
blocks_one_hot = blocks_one_hot.reshape(blocks_one_hot.shape[0], -1)
batch_graph.ndata["blocks"] = blocks_one_hot.float()
labels = batch_graph.ndata[f"{TARGET_SYMBOL}_labels"]

in_size = batch_graph.ndata["blocks"].shape[1]
out_size = int(max(labels) + 1)
hidden_size = int(np.median([in_size, out_size]))
model = MyConvolution(batch_graph, in_size, hidden_size, out_size)

index = np.array(range(batch_graph.num_nodes()))
# TODO: Dataset is EXTREMELY unbalanced. (Subsample 0 class?)
train_idx, test_idx = train_test_split(index, random_state=33, train_size=0.7, stratify=labels)

opt = t.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
best_test_acc = 0

loss_weights = t.full((int(labels.max()) + 1,), 1, dtype=t.float)
label_count = Counter(labels.numpy())
loss_weights[1] = label_count[0] / label_count[1]


if t.cuda.is_available():
    print("Going Cuda!")
    dev = t.device("cuda:0")
    batch_graph = batch_graph.to(dev)
    labels = batch_graph.ndata[f"{TARGET_SYMBOL}_labels"]
    model.cuda(dev)
    train_idx = t.tensor(train_idx, device=dev)
    test_idx = t.tensor(test_idx, device=dev)
    loss_weights = loss_weights.cuda(dev)

for epoch in range(EPOCHS):
    logits = model(batch_graph)
    loss = cross_entropy(logits[train_idx], labels[train_idx].long(), weight=loss_weights)

    pred = logits.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    if test_acc > best_test_acc:
        best_test_acc = test_acc

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 5 == 0:
        print(
            f"Loss {loss.item():.4f}, Train Acc {train_acc.item():.4f}, Test Acc {test_acc.item():.4f}"
            + f"(Best {best_test_acc.item():.4f})"
        )


with open(MODEL_PATH, "wb+") as f:
    pickle.dump(model, f)

print("Done.")
