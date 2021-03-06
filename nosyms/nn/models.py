from typing import List
from torch.nn import ModuleList, Module, Linear
from torch.nn import functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv


class BallConvolution(Module):
    def __init__(self, etypes: List[str], conv_layers: int, in_size: int, hidden_size: int, out_size: int):
        """
        Convolutional neural network which works with ball encoding.
        :param etypes: List of edge types to use in the convolutional layers
        :param conv_layers: Number of convolutional layers
        :param in_size: In size for classifier network
        :param hidden_size: Hidden size for classifier network
        :param out_size: Out size for classifier network
        """
        super(BallConvolution, self).__init__()
        self.training = True
        self.conv_layers = ModuleList(
            [
                HeteroGraphConv(
                    {
                        etype: GraphConv(in_size, in_size, norm="right", weight=True, activation=F.relu)
                        for etype in etypes
                    },
                    aggregate="mean",
                )
                for _ in range(conv_layers)
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
            result = layer(F.dropout(result, training=self.training))
        return result

    def forward_batch(self, blocks):
        if len(blocks) != len(self.conv_layers):
            raise ValueError(f"Len of blocks {len(blocks)} not matching num of conv layers: {len(self.conv_layers)}")
        h_dict = {"chunk": blocks[0].srcdata["blocks"]}
        for layer, bs in zip(self.conv_layers, blocks):
            h_dict = layer(bs, h_dict)
        result = h_dict["chunk"]
        for layer in self.class_layers:
            result = layer(F.dropout(result, training=self.training))
        return result
