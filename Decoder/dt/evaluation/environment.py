"""
Influence Maximization Environment.

This module implements the environment for simulating information spread on graphs.
It supports multiple propagation models (MC, RR) and provides reward computation
based on influence spread. The environment can operate in training or evaluation mode.

Key features:
- Supports Monte Carlo (MC) and Reverse Reachable (RR) set methods
- Caching mechanism for RR sets to accelerate computation
- Reward computation based on incremental influence spread
"""

import numpy as np
import statistics
from multiprocessing import Pool
import random
import time
import Decoder.utils.graph_utils as graph_utils

random.seed(123)
np.random.seed(123)


class Environment:
    """
    Environment for Influence Maximization (IM) problem.
    
    Simulates information propagation on graphs and computes rewards based on
    the number of nodes influenced by the selected seed set.
    """
    def __init__(self, name, graphs, budget, method='RR', use_cache=False, training=True):
        """
        Initialize the Influence Maximization environment.
        
        Args:
            name: Environment name (currently only 'IM' supported)
            graphs: List of graph objects to use
            budget: Maximum number of seed nodes to select
            method: Propagation model ('MC' for Monte Carlo, 'RR' for Reverse Reachable sets)
            use_cache: Whether to use caching for influence computation
            training: Whether in training mode (computes reward at each step) or evaluation mode (only at end)
        """
        self.name = name
        self.graphs = graphs
        self.budget = budget
        self.method = method
        self.use_cache = use_cache
        # Initialize caches for different methods
        if self.use_cache:
            if self.method == 'MC':
                self.influences = {}  # Cache for MC influence values
            elif self.method == 'RR':
                self.RRs_dict = {}  # Cache for RR sets (keyed by graph id)
        self.training = training

    def reset_graphs(self, num_graphs=10):
        raise NotImplementedError()

    def reset(self, idx=None, training=True):
        """
        Reset the environment for a new episode.
        
        Args:
            idx: Index of graph to use (None for random selection)
            training: Whether in training mode
            
        Returns:
            Initial state (all zeros, no nodes selected)
        """
        if idx is None:
            self.graph = random.choice(self.graphs)
        else:
            self.graph = self.graphs[idx]
        # Initialize state: 0 = not selected, 1 = selected
        self.state = [0 for _ in range(self.graph.num_nodes)]
        self.prev_inf = 0  # Previous influence value for incremental reward
        # Get or create RR cache for this graph
        if self.use_cache and self.method == 'RR':
            self.RRs = self.RRs_dict.setdefault(id(self.graph), [])
        self.states = []  # History of states
        self.actions = []  # History of selected nodes
        self.rewards = []  # History of rewards
        self.training = training

    def compute_reward(self, S):
        """
        Compute reward based on influence spread of seed set S.
        
        The reward is the incremental influence: current_influence - previous_influence.
        This encourages selecting nodes that maximize marginal gain.
        
        Args:
            S: List of selected seed nodes
            
        Returns:
            Reward (incremental influence spread)
        """
        num_process = 5  # Number of parallel processes for MC method
        num_trial = 10000  # Number of trials for influence estimation

        need_compute = True
        # Check if result is cached (for MC method)
        if self.use_cache and self.method == 'MC':
            S_str = f"{id(self.graph)}.{','.join(map(str, sorted(S)))}"
            need_compute = S_str not in self.influences

        if need_compute:
            if self.method == 'MC':
                # Monte Carlo: Forward propagation simulation with parallel processing
                with Pool(num_process) as p:
                    es_inf = statistics.mean(p.map(graph_utils.workerMC, 
                        [[self.graph, S, int(num_trial / num_process)] for _ in range(num_process)]))
            elif self.method == 'RR':
                # Reverse Reachable: Use cached RR sets if available
                if self.use_cache:
                    es_inf = graph_utils.computeRR(self.graph, S, num_trial, cache=self.RRs)
                else:
                    es_inf = graph_utils.computeRR(self.graph, S, num_trial)
            else:
                raise NotImplementedError(f'{self.method}')

            # Cache the result for MC method
            if self.use_cache and self.method == 'MC':
                self.influences[S_str] = es_inf
        else:
            es_inf = self.influences[S_str]

        # Compute incremental reward (marginal gain)
        reward = es_inf - self.prev_inf
        self.prev_inf = es_inf
        self.rewards.append(reward)

        return reward

    def step(self, node, time_reward=None):
        """
        Execute one step: select a node and update the environment.
        
        In training mode: computes reward at each step.
        In evaluation mode: only computes reward when budget is reached (done=True).
        This optimization significantly reduces computation time during evaluation.
        
        Args:
            node: Index of node to select as seed
            time_reward: Optional list to store computation time
            
        Returns:
            Tuple of (states_history, reward, done)
            - states_history: List of all states visited
            - reward: Incremental influence spread (None in eval mode if not done)
            - done: Whether budget is reached
        """
        # Skip if node already selected
        if self.state[node] == 1:
            return
        self.states.append(self.state.copy())
        self.actions.append(node)
        self.state[node] = 1  # Mark node as selected
        if self.name != 'IM':
            raise NotImplementedError(f'Environment {self.name}')

        S = self.actions  # Current seed set
        done = len(S) >= self.budget  # Check if budget is reached

        # Reward computation strategy:
        # Training: compute at every step (for learning)
        # Evaluation: compute only at the end (for efficiency)
        if self.training:
            reward = self.compute_reward(S)
        else:
            if done:
                if time_reward is not None:
                    start_time = time.time()
                reward = self.compute_reward(S)
                if time_reward is not None:
                    time_reward[0] = time.time() - start_time
            else:
                reward = None  # Skip computation during episode

        return (self.states, reward, done)
