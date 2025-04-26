import os
import pickle
import torch
from torch_geometric.data import Data

def load_graphs_from_gpickle(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".gpickle")]
    all_graphs = []

    node_type_vocab = {}  # mapping of nodeType to index

    # First pass: build vocab of all node labels
    for filename in files:
        with open(os.path.join(folder, filename), "rb") as f:
            g = pickle.load(f)
        for _, attr in g.nodes(data=True):
            label = attr.get("label", "Unknown")
            if label not in node_type_vocab:
                node_type_vocab[label] = len(node_type_vocab)

    num_node_types = len(node_type_vocab)

    # Second pass: convert each graph
    for filename in files:
        with open(os.path.join(folder, filename), "rb") as f:
            g = pickle.load(f)

        # build features & mapping
        node_features = []
        node_id_map = {}
        for i, (node_id, attr) in enumerate(g.nodes(data=True)):
            label = attr.get("label", "Unknown")
            idx = node_type_vocab[label]

            # one-hot vector of length = num_node_types
            oh = [0] * num_node_types
            oh[idx] = 1
            node_features.append(oh)

            node_id_map[node_id] = i

        # now build PyG Tensor
        x = torch.tensor(node_features, dtype=torch.float)            # [num_nodes, num_node_types]

        # edges → edge_index
        edges = [
            (node_id_map[u], node_id_map[v])
            for u, v in g.edges()
            if u in node_id_map and v in node_id_map
        ]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # graph label
        y = torch.tensor([g.graph["label"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        all_graphs.append(data)

    print(f"Loaded {len(all_graphs)} graphs with {num_node_types} unique node types.")
    return all_graphs, num_node_types
