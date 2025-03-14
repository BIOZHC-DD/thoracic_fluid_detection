import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, global_mean_pool


class variant2(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, output_dim, num_heads=8):
        super(variant2, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_attention = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.edge_mlp(edge_attr)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = F.elu(self.conv1(x, edge_index))

        # Edge attention
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        edge_attention_weights = F.softmax(self.edge_attention(edge_features).squeeze(), dim=0)

        # Aggregate edge features with attention
        aggregated_edge_features = torch.zeros_like(x)
        for i in range(edge_attr.size(0)):
            aggregated_edge_features[row[i]] += edge_attention_weights[i] * edge_attr[i]

        # Combinining features
        x = x + aggregated_edge_features

        # Second GAT layer
        x = F.elu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)

        out = self.fc(x)
        out = out.view(-1)
        return out


class variant3(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_heads=1):
        super(variant3, self).__init__()

        # First GAT layer
        self.conv1 = GATConv(num_node_features, hidden_dim // num_heads, heads=num_heads)

        # Second GAT layer
        #self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)

        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Add self-loops to edge indices
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # First GAT layer with ELU activation
        x = F.elu(self.conv1(x, edge_index))

        # Second GAT layer with ELU activation
        #x = F.elu(self.conv2(x, edge_index))

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Final prediction
        out = self.fc(x)
        out = out.view(-1)

        return out

class variant1(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, output_dim, num_heads=1):
        super(variant1, self).__init__()

        self.conv1 = GATConv(num_node_features, hidden_dim // num_heads, heads=num_heads)

        # Edge processing
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge attention based on edge attributes
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Process edge features
        processed_edge_attr = self.edge_mlp(edge_attr)

        # Compute attention weights directly from edge attributes
        edge_attention_weights = F.softmax(self.edge_attention(processed_edge_attr).squeeze(), dim=0)

        # First GAT layer
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = F.elu(self.conv1(x, edge_index_with_self_loops))

        # Aggregate edge features with attention
        row, col = edge_index
        aggregated_edge_features = torch.zeros_like(x)
        for i in range(edge_attr.size(0)):
            aggregated_edge_features[row[i]] += edge_attention_weights[i] * processed_edge_attr[i]

        # Combine node and edge features
        x = x + aggregated_edge_features

        # Global pooling and final prediction
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        out = out.view(-1)
        return out


class variant0(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, output_dim, num_heads=1):
        super(variant0, self).__init__()

        self.conv1 = GATConv(num_node_features, hidden_dim // num_heads, heads=num_heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)

        # Edge processing
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Edge attention based on edge attributes
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # Process edge features
        processed_edge_attr = self.edge_mlp(edge_attr)

        # Compute attention weights directly from edge attributes
        edge_attention_weights = F.softmax(self.edge_attention(processed_edge_attr).squeeze(), dim=0)

        # First GAT layer
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = F.elu(self.conv1(x, edge_index_with_self_loops))

        # Aggregate edge features with attention
        row, col = edge_index
        aggregated_edge_features = torch.zeros_like(x)
        for i in range(edge_attr.size(0)):
            aggregated_edge_features[row[i]] += edge_attention_weights[i] * processed_edge_attr[i]

        # Combine node and edge features
        x = x + aggregated_edge_features

        # Second GAT layer
        x = F.elu(self.conv2(x, edge_index_with_self_loops))

        # Global pooling and final prediction
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        out = out.view(-1)
        return out