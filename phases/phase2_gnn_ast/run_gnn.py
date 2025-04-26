# -- Final Full GNN Trainer --

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model_gin import GINNet
from utils.load_graphs import load_graphs_from_gpickle

# Load graphs
print("📦 Loading graphs...")
graphs, num_node_types = load_graphs_from_gpickle("data/graphs/gnn_graphs")

# Split into Train/Val/Test
train_graphs, test_graphs = train_test_split(
    graphs, test_size=0.2, random_state=42, stratify=[g.y.item() for g in graphs]
)
train_graphs, val_graphs = train_test_split(
    train_graphs, test_size=0.1, random_state=42, stratify=[g.y.item() for g in train_graphs]
)

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=64)
test_loader = DataLoader(test_graphs, batch_size=64)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINNet(num_node_features=num_node_types, hidden_channels=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Weighted loss for class imbalance
from collections import Counter
label_counter = Counter([data.y.item() for data in train_graphs])
total_samples = sum(label_counter.values())
weights = torch.tensor([
    total_samples / label_counter[0],
    total_samples / label_counter[1],
], dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)

# Training loop
def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)
    return total_loss / len(train_loader), correct / total

# Evaluation loop
def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            y_true.extend(batch.y.tolist())
            y_pred.extend(preds.tolist())
    return classification_report(y_true, y_pred, target_names=["Secure", "Vulnerable"])

# Training phase
print("🚀 Training...")
for epoch in range(1, 201):
    loss, acc = train()
    print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Accuracy: {acc:.4f}")

# Final evaluation
print("\n📊 Validation Results:")
print(evaluate(val_loader))
print("\n🧪 Test Results:")
print(evaluate(test_loader))
