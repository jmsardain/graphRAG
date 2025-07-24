import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train_gnn_and_get_embeddings(graph, epochs=50, hidden_dim=64, out_dim=32):
    # Convert NetworkX graph to PyG Data object
    pyg_graph = from_networkx(graph)
    # Assume node features are stored as 'feature' attribute
    x = []
    for i in range(pyg_graph.num_nodes):
        node = pyg_graph._node[i]
        x.append(node['feature'])
    x = torch.tensor(np.stack(x), dtype=torch.float)
    edge_index = pyg_graph.edge_index

    model = GCN(x.size(1), hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        # Dummy loss: unsupervised, so use feature reconstruction (autoencoder style)
        loss = F.mse_loss(out, x[:, :out_dim])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index).cpu().numpy()

    # Map embeddings back to node names
    node_names = list(graph.nodes)
    node_emb_dict = {name: embeddings[i] for i, name in enumerate(node_names)}
    return node_emb_dict 