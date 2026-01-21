"""
Action Trainer for Behavior Cloning (BC) Models.

This module implements the training loop for simple MLP-based
Behavior Cloning models. BC models predict actions directly from states
without using sequence modeling.
"""

import numpy as np
import torch

from Decoder.dt.training.trainer import Trainer


class ActTrainer(Trainer):
    """
    Trainer for Behavior Cloning (BC) models.
    
    BC models use a simple MLP to predict actions from states.
    Training uses MSE loss between predicted and expert actions.
    """

    def train_step(self):
        """
        Perform one training step for BC model.
        
        Steps:
        1. Sample batch of trajectories
        2. Forward pass (predict action from state)
        3. Compute MSE loss on action predictions
        4. Backward pass and update
        
        Returns:
            Training loss value
        """
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:,0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:,-1].reshape(-1, act_dim)

        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target, action_target, reward_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
