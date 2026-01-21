"""
Diffusion-Aware Transformer (DAT) for Graph Representation Learning.

This module implements the encoder part of the PIM architecture:
- Graph Attention Network (GAT) for initial node embeddings
- Graph Transformer layers for message passing and feature aggregation
- Support for edge features and positional encodings
- Outputs node and edge embeddings for downstream tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.nn import Linear

from Encoder.DAT.layers import GraphTransformerLayer
from Encoder.DAT.layers import MLPReadout


class GraphTransformerNet(nn.Module):
    """
    Graph Transformer Network for learning graph representations.
    
    Architecture:
    1. GAT layer: Initial node embedding with graph attention
    2. Positional encoding: Laplacian or WL positional encoding
    3. Graph Transformer layers: Multi-layer message passing
    4. Output: Node and edge embeddings
    
    This encoder transforms raw graph data into learned embeddings that
    capture structural and semantic information for influence maximization.
    """
    def __init__(self, net_params):
        """
        Initialize Graph Transformer Network.
        
        Args:
            net_params: Dictionary containing network hyperparameters:
                - node_in_dim: Input node feature dimension
                - edge_in_dim: Input edge feature dimension
                - hidden_dim: Hidden layer dimension
                - n_heads: Number of attention heads
                - out_dim: Output dimension
                - L: Number of transformer layers
                - lap_pos_enc: Whether to use Laplacian positional encoding
                - edge_feat: Whether to use edge features
        """
        super().__init__()
        node_in_dim = net_params['node_in_dim']
        edge_in_dim = net_params['edge_in_dim']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        # Initial GAT layer for node embedding
        self.gat_layer = GATConv(in_channels=node_in_dim, out_channels=hidden_dim, heads=1, concat=True)

        # Edge feature processing
        if self.edge_feat:
            self.edge_linear = Linear(in_features=edge_in_dim, out_features=hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # Stack of Graph Transformer layers
        # All but last layer: hidden_dim -> hidden_dim
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        # Last layer: hidden_dim -> out_dim
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.MLP_layer = MLPReadout(out_dim, 1)  # Final readout layer

    def forward(self, g, h_lap_pos_enc=None):
        """
        Forward pass through Graph Transformer.
        
        Process:
        1. Initial GAT embedding
        2. Add positional encoding
        3. Process edge features
        4. Pass through transformer layers
        5. Return node and edge embeddings
        
        Args:
            g: Graph data object (PyTorch Geometric Data)
            h_lap_pos_enc: Laplacian positional encoding [num_nodes, pos_enc_dim]
            
        Returns:
            Tuple of (node_embeddings, edge_embeddings, edge_index)
            - h: Node embeddings [num_nodes, out_dim]
            - e: Edge embeddings [num_edges, hidden_dim]
            - edge_index: Edge connectivity [2, num_edges]
        """
        device = torch.device('cuda:0')
        # Move all tensors to device
        if g.edge_index.device != device:
            g.edge_index = g.edge_index.to(device)
        if g.edge_weight.device != device:
            g.edge_weight = g.edge_weight.to(device)
        if g.lap_pos_enc.device != device:
            g.lap_pos_enc = g.lap_pos_enc.to(device)
        g.features = torch.tensor(g.features)
        if g.features.device != device:
            g.features = g.features.to(device)
        g.features = g.features.to(dtype=torch.float32)
        g.lap_pos_enc = g.lap_pos_enc.to(dtype=torch.float64)
        
        # Initial node embedding using GAT
        h = self.gat_layer(x=g.features, edge_index=g.edge_index)
        h = self.in_feat_dropout(h)
        
        # Add Laplacian positional encoding
        if self.lap_pos_enc:
            if h_lap_pos_enc.device != device:
                h_lap_pos_enc = h_lap_pos_enc.to(device)
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc  # Add positional encoding to node features

        # Process edge features
        if not self.edge_feat:
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.edge_linear(g.edge_weight)

        # Pass through Graph Transformer layers
        for conv in self.layers:
            h, e = conv(g, h, e)  # Update node and edge embeddings
            edge_index = g.edge_index
        
        # Return last node embedding (for readout)
        h = h[:,-1]
        return h, e, edge_index

