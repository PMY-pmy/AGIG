"""
Behavior Cloning (BC) Model.

This module implements a simple MLP-based model for behavior cloning.
Unlike Decision Transformer, BC models do not use sequence modeling
and directly predict actions from concatenated states.
"""

import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):
    """
    Base class for trajectory models.
    
    Provides common interface for models that process state-action-reward sequences.
    """

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])


class MLPBCModel(TrajectoryModel):
    """
    Behavior Cloning Model using Multi-Layer Perceptron (MLP).
    
    Simple feedforward network that predicts next action from concatenated states.
    Unlike Decision Transformer, this model does not use:
    - Sequence modeling (transformer)
    - Return-to-go conditioning
    - Attention mechanisms
    
    It simply concatenates the last max_length states and predicts the action.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        """
        Initialize MLP Behavior Cloning model.
        
        Args:
            state_dim: Dimension of state vector
            act_dim: Dimension of action vector
            hidden_size: Hidden layer size
            n_layer: Number of hidden layers
            dropout: Dropout rate
            max_length: Number of past states to concatenate
        """
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length

        # Build MLP: concatenate states -> hidden layers -> action prediction
        layers = [nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),  # Tanh activation for action bounds
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):
        """
        Forward pass: predict action from concatenated states.
        
        Args:
            states: State sequences [batch, seq_len, state_dim]
            actions: Not used (for interface compatibility)
            rewards: Not used
            attention_mask: Not used
            target_return: Not used (BC doesn't use RTG)
            
        Returns:
            Tuple of (None, action_preds, None)
            - action_preds: Predicted actions [batch, 1, act_dim]
        """
        # Concatenate last max_length states
        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states
        # Predict action from concatenated states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    def get_action(self, states, actions, rewards, **kwargs):
        """
        Get action prediction during inference.
        
        Args:
            states: History of states [seq_len, state_dim]
            actions: Not used
            rewards: Not used
            
        Returns:
            Predicted action vector [act_dim]
        """
        states = states.reshape(1, -1, self.state_dim)
        # Pad if sequence is shorter than max_length
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]
