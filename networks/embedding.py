from torch.nn import ModuleList, Module, Linear
from torch.nn import functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
from hyperparams import BALL_CONV_LAYERS


class MyConvolution(Module):
    def __init__(self, graph, in_size, hidden_size, out_size):
        super(MyConvolution, self).__init__()
        self.conv_layers = ModuleList(
            [
                HeteroGraphConv(
                    {
                        etype: GraphConv(in_size, in_size, norm="right", weight=True, activation=F.relu)
                        for etype in graph.etypes
                    },
                    aggregate="mean",
                )
                for _ in range(BALL_CONV_LAYERS)
            ]
        )
        sizes = [in_size, hidden_size, out_size]
        self.class_layers = ModuleList([Linear(i, o) for i, o in zip(sizes, sizes[1:])])

    def forward(self, graph):
        h_dict = {"chunk": graph.ndata["blocks"]}
        for layer in self.conv_layers:
            h_dict = layer(graph, h_dict)
        result = h_dict["chunk"]
        for layer in self.class_layers:
            result = layer(result)
        return result

    def forward_batch(self, blocks):
        if len(blocks) != len(self.conv_layers):
            raise ValueError(f"Len of blocks {len(blocks)} not matching num of conv layers: {len(self.conv_layers)}")
        h_dict = {"chunk": blocks[0].srcdata["blocks"]}
        for layer, blocks in zip(self.conv_layers, blocks):
            h_dict = layer(blocks, h_dict)
        result = h_dict["chunk"]
        for layer in self.class_layers:
            result = layer(result)
        return result
