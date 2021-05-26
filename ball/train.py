import pickle
import warnings
import numpy as np
import torch as t
import torch.cuda
from torch.nn.functional import cross_entropy, one_hot
from sklearn.metrics import rand_score
from sklearn.model_selection import train_test_split
import dgl
from networks.embedding import MyConvolution

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")
warnings.filterwarnings("ignore", message="Undefined\ type\ encountered")

with open("./ball-sym-data.pkl", "rb") as f:
    all_data = pickle.load(f)

TARGET_SYMBOL = "task_struct"

rands = []
for path, graph, node_ids in all_data:
    true_labels = np.zeros(graph.num_nodes())
    for i, n in node_ids.inv.items():
        if TARGET_SYMBOL in n.type_descriptor:  # TODO This is not he safest check, but it works for task_struct :)
            true_labels[i] = n.chunk
    graph.ndata[f"{TARGET_SYMBOL}_labels"] = t.tensor(true_labels)
    encodings = {}
    encoding_labels = np.zeros(graph.num_nodes())
    for i, bs in enumerate(graph.ndata["blocks"]):
        key = tuple(bs.numpy())
        if key not in encodings:
            encodings[key] = len(encodings)
        encoding_labels[i] = encodings[key]
    # Many balls end up with identical type encodings (not necessarily identical edges, tho)
    rs = rand_score(true_labels, encoding_labels)
    rands.append(rs)


print(
    f"""Rand score for encodings:
mean: {np.mean(rands)}
std:  {np.std(rands)}"""
)

batch_graph = dgl.batch([g for _, g, _ in all_data])

if t.cuda.is_available():
    # GPU can't handle that many features. :(
    batch_graph.ndata["blocks"] = batch_graph.ndata["blocks"][:, 80:-80]

# We do this here rather than in the original encoding for for storage efficiency
blocks_one_hot = one_hot(batch_graph.ndata["blocks"].long())
blocks_one_hot = blocks_one_hot.reshape(blocks_one_hot.shape[0], -1)
batch_graph.ndata["blocks"] = blocks_one_hot.float()
labels = batch_graph.ndata[f"{TARGET_SYMBOL}_labels"]

in_size = batch_graph.ndata["blocks"].shape[1]
out_size = int(max(labels) + 1)
hidden_size = int(np.median([in_size, out_size]))
model = MyConvolution(batch_graph, in_size, in_size, out_size)

index = np.array(range(batch_graph.num_nodes()))
# TODO: Dataset is EXTREMELY unbalanced. (Subsample 0 class?)
train_idx, test_idx = train_test_split(index, random_state=33, train_size=0.7, stratify=labels)


opt = t.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
best_test_acc = 0

loss_weights = t.zeros(int(labels.max()) + 1) + 1
loss_weights[0] = 200

if t.cuda.is_available():
    print("Going Cuda!")
    dev = t.device("cuda:0")
    batch_graph = batch_graph.to(dev)
    labels = batch_graph.ndata[f"{TARGET_SYMBOL}_labels"]
    model.cuda(dev)
    train_idx = t.tensor(train_idx, device=dev)
    test_idx = t.tensor(test_idx, device=dev)
    loss_weights = loss_weights.cuda(dev)

for epoch in range(100):
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


with open("./model.pkl", "wb+") as f:
    pickle.dump(model, f)

print("Done.")
