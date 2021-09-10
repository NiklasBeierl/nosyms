import json

import networkx as nx
import pandas as pd

from paging_detection import Snapshot, PageTypes


def get_node_features(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Create a pandas dataframe with some useful stats for every node in a paging structures graph.
    """
    df = pd.DataFrame(data=[graph.nodes[node] for node in graph.nodes], index=graph.nodes)
    idx, deg = tuple(zip(*graph.in_degree))
    df["in_degree"] = pd.Series(data=deg, index=idx)
    idx, deg = tuple(zip(*graph.out_degree))
    # Note: Out degree will be 0 for page tables in "training" data and >0 in "target" data
    df["out_degree"] = pd.Series(data=deg, index=idx)
    return df


if __name__ == "__main__":
    with open("../data_dump/all_pages.json") as f:
        snapshot = Snapshot.validate(json.load(f))

    pages = snapshot.pages

    graph = nx.read_graphml("../data_dump/all_pages.graphml")
    data_pages = [node for node, data in graph.nodes.items() if not data[str(PageTypes.PT)]]
    graph.remove_nodes_from(data_pages)

    node_data = get_node_features(graph)

print("Done")
