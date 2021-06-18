import pickle
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
from torch.nn.functional import softmax
from file_paths import MEM_GRAPH_PATH, RESULTS_PATH, CONF_MAT_PATH, ROC_PATH
import develop.filter_warnings

with open(MEM_GRAPH_PATH, "rb") as f:
    mem_graph = pickle.load(f)

with open(RESULTS_PATH, "rb") as f:
    results = pickle.load(f)


post_results = softmax(results.cpu(), dim=1)
post_results = post_results.detach().numpy()

labels = (mem_graph.ndata["pids"] != -1).numpy()
ras = roc_auc_score(labels, post_results[:, 1])

yay = post_results[mem_graph.ndata["pids"] != -1, 1]
yay_mean = yay.mean()
yay_std = yay.std()
nay = post_results[mem_graph.ndata["pids"] == -1, 0].mean()
nay_mean = nay.mean()
nay_std = nay.std()

post_results_am = post_results.argmax(1)

pids = set(mem_graph.ndata["pids"].numpy())
task_results = [np.array(post_results_am[mem_graph.ndata["pids"] == pid]) for pid in sorted(pids) if pid != -1]
length = max(len(tr) for tr in task_results)
task_results = [np.resize(tr, length) for tr in task_results]
task_results = np.stack(task_results)


m_confusion = confusion_matrix(labels, post_results_am, normalize="true")
m_recall = recall_score(labels, post_results_am, average="macro")
m_prec = precision_score(labels, post_results_am, average="macro")
m_f1 = f1_score(labels, post_results_am, average="macro")


cf_cols = ["not task_struct", "task_struct"]
df_cm = pd.DataFrame(m_confusion, columns=cf_cols, index=cf_cols)
df_cm.index.name = "Actual"
df_cm.columns.name = "Predicted"
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.5)
sn.heatmap(
    df_cm, cmap="Blues", annot=True, fmt=".2%", annot_kws={"size": 20}, yticklabels=False, xticklabels=False, cbar=False
)
plt.yticks(np.arange(2) + 0.5, cf_cols, rotation=90, fontsize=15, va="center")
plt.xticks(np.arange(2) + 0.5, cf_cols, rotation=0, fontsize=15, va="center")
# plt.show()
plt.savefig(CONF_MAT_PATH)

from plot_metric.functions import BinaryClassification

bc = BinaryClassification(labels, post_results[:, 1], labels=cf_cols)
plt.figure(figsize=(10, 7))
bc.plot_roc_curve()
# plt.show()
plt.savefig(ROC_PATH)

# Some of the outputs can only be inspected when running interactively. :)
print("Done")
