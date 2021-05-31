import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from torch.nn.functional import softmax
from file_paths import MEM_GRAPH_PATH, RESULTS_PATH


import warnings

warnings.filterwarnings("ignore", message="DGLGraph\.__len__")

with open(MEM_GRAPH_PATH, "rb") as f:
    mem_graph = pickle.load(f)

with open(RESULTS_PATH, "rb") as f:
    results = pickle.load(f)


post_results = softmax(results.cpu(), dim=1)
post_results = post_results.detach().numpy()

yay = post_results[mem_graph.ndata["pids"] != -1, 1]
yay_mean = yay.mean()
yay_std = yay.std()
nay = post_results[mem_graph.ndata["pids"] == -1, 0].mean()
nay_mean = nay.mean()
nay_std = nay.std()

post_results = post_results.argmax(1)
pids = set(mem_graph.ndata["pids"].numpy())
task_results = [np.array(post_results[mem_graph.ndata["pids"] == pid]) for pid in sorted(pids) if pid != -1]
length = max(len(tr) for tr in task_results)
task_results = [np.resize(tr, length) for tr in task_results]
task_results = np.stack(task_results)

labels = (mem_graph.ndata["pids"] != -1).numpy()


m_confusion = confusion_matrix(labels, post_results, normalize="true")
m_recall = recall_score(labels, post_results, average="macro")
m_prec = precision_score(labels, post_results, average="macro")
m_f1 = f1_score(labels, post_results, average="macro")

# This file only really makes sense if used in a debugger with facilities to properly visualise tensors / arrays.
print("Done")
