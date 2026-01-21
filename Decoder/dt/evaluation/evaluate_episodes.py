"""
Episode Evaluation Functions.

This module provides functions for evaluating trained models on influence maximization tasks.
It simulates episodes where the model selects seed nodes sequentially, and computes
the final influence spread as the evaluation metric.

Key functions:
- evaluate_episode_rtg: Evaluation with Return-to-Go (RTG) for Decision Transformer
- evaluate_episode: Evaluation for Behavior Cloning models
"""

import numpy as np
import torch
import Decoder.dt.evaluation.models as models
import random
from Encoder.graph import MyGT_gen
from torch_geometric.data import Data

def select_action(graph, state, epsilon, training=True, budget=None, device='cuda'):
    """
    Select action using epsilon-greedy policy (for training).
    
    Args:
        graph: Graph object
        state: Current state vector
        epsilon: Exploration rate
        training: Whether in training mode
        budget: Budget constraint
        device: Computing device
        
    Returns:
        Selected node index
    """
        if not(training):
            graph_input = MyGT_gen(graph)
            with torch.no_grad():
                q_a = MyGT_gen(graph_input)
            q_a[state.nonzero()] = -1e5

            if budget is None:
                return torch.argmax(q_a).detach().clone()
            else:
                return torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone()
        # training
        available = (state == 0).nonzero()
        if epsilon > random.random():
            return random.choice(available)
        else:
            graph_input_h, graph_input_e = MyGT_gen(graph)
            graph_input = Data(x=graph_input_h, edge_index=graph_input_e)

            with torch.no_grad():
                model = models.predict_action().to(device=device)
                q_a = model(graph_input)
            max_position = (q_a == q_a[available].max().item()).nonzero()
            return torch.tensor(
                [random.choice(
                    np.intersect1d(available.cpu().contiguous().view(-1).numpy(),
                        max_position.cpu().contiguous().view(-1).numpy()))],
                dtype=torch.long)

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=6,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):
    """
    Evaluate Behavior Cloning model on a single episode.
    
    Simpler evaluation function for BC models that don't use RTG.
    The model predicts actions directly from states without conditioning
    on return-to-go values.
    
    Args:
        env: Influence maximization environment
        state_dim: Dimension of state space
        act_dim: Dimension of action space
        model: Trained BC model
        max_ep_len: Maximum episode length
        device: Computing device
        target_return: Not used for BC (for interface compatibility)
        mode: Evaluation mode
        state_mean: Mean for state normalization
        state_std: Std for state normalization
        
    Returns:
        Tuple of (episode_return, episode_length)
    """
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # Initialize sequences
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    # Main episode loop
    for t in range(max_ep_len):

        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=31,
        scale=1.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    """
    Evaluate Decision Transformer model on a single episode using Return-to-Go (RTG).
    
    This function simulates an episode where:
    1. Model predicts next action given current state and target RTG
    2. Action is executed in the environment
    3. RTG is updated based on received reward
    4. Process repeats until budget is reached
    
    The RTG mechanism enables personalization: different target spreads (TS)
    result in different RTG values, leading to different seed selection strategies.
    
    Args:
        env: Influence maximization environment
        state_dim: Dimension of state space (number of nodes)
        act_dim: Dimension of action space (number of nodes)
        model: Trained Decision Transformer model
        max_ep_len: Maximum episode length (budget * num_nodes)
        scale: Scaling factor for RTG normalization
        state_mean: Mean for state normalization
        state_std: Std for state normalization
        device: Computing device
        target_return: Initial return-to-go value (target influence spread)
        mode: Evaluation mode ('normal', 'noise', 'delayed')
        
    Returns:
        Tuple of (episode_return, episode_length, final_influence)
        - episode_return: Cumulative reward (influence spread)
        - episode_length: Number of steps taken
        - final_influence: Final influence value R
    """
    model.eval()
    model.to(device=device)

    env.reset()
    state = env.state
    state = np.array(state)
    graph = env.graph
    state_dim = graph.num_nodes
    act_dim = graph.num_nodes

    states = []
    actions = []
    rewards = []
    if mode == 'noise':
        state = env.state + np.random.normal(0, 0.1, size=env.state.shape)

    # Initialize sequences
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    # Initialize return-to-go (RTG) for personalization
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []
    action_index_list = []  # Track selected nodes to avoid duplicates
    episode_return, episode_length = 0, 0
    
    # Main episode loop: select nodes until budget is reached
    for t in range(max_ep_len):

        # Prepare sequences for model input
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        # Get action prediction from model (conditioned on RTG for personalization)
        action = model.get_action(
            (states.to(dtype=torch.float32).cpu() - state_mean) / state_std,  # Normalized states
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),  # RTG enables personalization
            timesteps.to(dtype=torch.long),
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        # Select node with highest probability that hasn't been selected yet
        for idx in np.argsort(action)[::-1]:
            if idx not in action_index_list:
                action_index = idx
                break

        action_index_list.append(action_index)
        # Execute action in environment (only computes reward when done in eval mode)
        reward, R, done = env.step(action_index)

        # Update state: mark selected node
        state[action_index] = 1
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # Update return-to-go: subtract received reward (unless in delayed mode)
        if mode != 'delayed':
            pred_return = target_return[0,-1] - reward  # RTG decreases as we get rewards
        else:
            pred_return = target_return[0,-1]  # RTG stays constant in delayed mode
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length, R
