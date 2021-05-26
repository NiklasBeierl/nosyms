import pickle
import torch
from torch.nn import functional as F
import dgl
import numpy as np
from encoding import BlockType
from encoding.sandwich import NODE_MAX_LEN

import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

with open("all_syms.pkl", "rb") as f:
    all_data = pickle.load(f)

target_symbol = "task_struct"

for _, graph, node_ids in all_data:
    # TODO: That check is not the "proper way"!
    relevant_nodes = [
        (node_id, dgl_id) for node_id, dgl_id in node_ids.items() if f'"{target_symbol}"' in node_id.type_descriptor
    ]
    relevant_nodes = sorted(relevant_nodes, key=lambda item: item[0].chunk)
    relevant_nodes = [dgl_id for _, dgl_id in relevant_nodes]

    target_labels = torch.zeros(graph.num_nodes())
    for i, dgl_id in enumerate(relevant_nodes):
        target_labels[dgl_id] = i + 1
    graph.ndata[f"{target_symbol}_labels"] = target_labels

batch_graph = dgl.batch([g for _, g, _ in all_data])

# TODO: Need to draw uniformly per label.
s1, s2 = int(0.5 * batch_graph.num_nodes()), int(0.75 * batch_graph.num_nodes())
shuffle = np.random.permutation(batch_graph.num_nodes())
train_idx = torch.tensor(shuffle[:s1]).long()
val_idx = torch.tensor(shuffle[s1:s2]).long()
test_idx = torch.tensor(shuffle[s2:]).long()
labels = batch_graph.ndata[f"{target_symbol}_labels"]

from networks.embedding import MyConvolution

in_size = NODE_MAX_LEN * len(BlockType)
out_size = int(max(labels) + 1)
hidden_size = int(np.median([in_size, out_size]))
model = MyConvolution(batch_graph, hidden_size, out_size)

opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = 0
best_test_acc = 0

try:
    for epoch in range(10):
        logits = model(batch_graph)
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].long())

        pred = logits.argmax(1)
        train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

        if best_val_acc <= val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            print(
                "Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)"
                % (
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                )
            )
except KeyboardInterrupt:
    pass  # Prematurely end training. Not pretty! I know!

with open("./model.pkl", "wb+") as f:
    torch.save(model, f)

print("Done")
