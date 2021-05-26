from torch import nn
from torch.nn import functional as F
import dgl.nn.pytorch as dglnn


class MyConvolution(nn.Module):
    def __init__(self, graph, in_size, hidden_size, out_size):
        super(MyConvolution, self).__init__()
        self.layer1 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(in_size, hidden_size, norm="none", weight=True) for etype in graph.etypes},
            aggregate="mean",
        )
        self.layer2 = dglnn.HeteroGraphConv(
            {etype: dglnn.GraphConv(hidden_size, out_size, norm="none", weight=True) for etype in graph.etypes},
            aggregate="mean",
        )

    def forward(self, graph, feature="blocks"):
        h_dict = self.layer1(graph, {"chunk": graph.ndata[feature]})
        h_dict = self.layer2(graph, h_dict)
        slice_classes = F.softmax(h_dict["chunk"])
        return slice_classes
