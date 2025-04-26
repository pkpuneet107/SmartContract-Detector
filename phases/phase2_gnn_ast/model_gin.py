import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GINNet(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GINNet, self).__init__()

        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.gin1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(0.5)

        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.gin2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout2 = nn.Dropout(0.5)

        self.classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.gin1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.gin2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)
