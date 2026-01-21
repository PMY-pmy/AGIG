"""
Sequence Trainer for Decision Transformer.

This module implements the training loop for Decision Transformer models.
It uses behavior cloning with MSE loss to learn action sequences conditioned
on return-to-go (RTG) values for personalization.
"""

import numpy as np
import torch

from Decoder.dt.training.trainer import Trainer


class SequenceTrainer(Trainer):
    """
    Trainer for Decision Transformer (sequence-to-sequence model).
    
    Trains the model to predict actions given states and return-to-go values.
    Uses Mean Squared Error (MSE) loss between predicted and target actions.
    """

    def train_step(self):
        """
        Perform one training step.
        
        Steps:
        1. Sample a batch of trajectory sequences
        2. Forward pass through the model
        3. Compute MSE loss on action predictions
        4. Backward pass and gradient update
        5. Apply gradient clipping for stability
        
        Returns:
            Training loss value
        """
        # Get batch of (state, action, reward, RTG) sequences
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        # Forward pass: predict next actions given states and RTG
        # RTG[:,:-1] because we predict action at time t given RTG at time t
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        # Flatten predictions and targets, mask out padding tokens
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # Compute MSE loss (only on actions, not states or rewards)
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)  # Gradient clipping for stability
        self.optimizer.step()

        # Track training metrics
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
