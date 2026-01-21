"""
Graph Utilities for Influence Maximization.

This module provides:
1. Graph data structure and operations
2. Influence computation methods (MC, RR, IC, SIS, LT)
3. Graph loading and preprocessing functions

Key methods:
- computeMC: Monte Carlo simulation for influence estimation
- computeRR: Reverse Reachable set method with caching support
- computeIC: Independent Cascade model simulation
- computeSIS: Susceptible-Infected-Susceptible model simulation
- computeLT: Linear Threshold model simulation
"""

import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from torch_geometric.utils import degree
import torch

random.seed(123)
np.random.seed(123)


class Graph:
    """
    Graph data structure for influence maximization.
    
    Stores nodes, edges, and adjacency information in formats optimized
    for influence computation algorithms.
    """
    def __init__(self, nodes, edges, children, parents, features):
        self.nodes = nodes # set()
        self.edges = edges # dict{(src,dst): weight, }
        self.children = children # dict{node: set(), }
        self.parents = parents # dict{node: set(), }
        self.features = features
        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])

    def get_prob(self, edge):
        return self.edges[edge]

    def get_adj(self):
        ''' return scipy sparse matrix '''
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge] # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

    def from_to_edges(self):
        ''' return a list of edge of (src,dst) '''
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        ''' return a list of edge of (src, dst) with edge weight '''
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight


def read_graph(path, ind=0, directed=False):
    """
    Load graph from edge list file.
    
    Reads graph data from a text file where each line contains two node indices.
    Constructs adjacency lists (children/parents) and computes edge weights.
    
    Args:
        path: Path to graph file (edge list format)
        ind: Index offset (0 or 1, depending on whether nodes start from 0 or 1)
        directed: Whether graph is directed (False for undirected)
        
    Returns:
        Graph object with nodes, edges, children, parents, and features
    """
    parents = {}
    children = {}
    edges = {}
    nodes = set()
    features = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split()
            src = int(row[0]) - ind
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst)
            parents.setdefault(dst, set()).add(src)
            edges[(src, dst)] = 0.0
            if not(directed):
                # regard as undirectional
                children.setdefault(dst, set()).add(src)
                parents.setdefault(src, set()).add(dst)
                edges[(dst, src)] = 0.0

    # Compute edge weights: 1 / in-degree (uniform probability)
    for src, dst in edges:
        edges[(src, dst)] = 1.0 / len(parents[dst])
    features = np.random.rand(len(nodes), 3)

    return Graph(nodes, edges, children, parents, features)

def computeMC(graph, S, R):
    """
    Compute influence spread using Monte Carlo (MC) simulation.
    
    Forward propagation method: starts from seed set S and simulates
    information spread R times, then averages the results.
    
    Algorithm:
    1. Start from seed set S
    2. For each active node, try to activate neighbors with probability = edge weight
    3. Repeat until no new activations
    4. Average over R simulations
    
    Args:
        graph: Graph object
        S: Seed set (list of node indices)
        R: Number of Monte Carlo simulations
        
    Returns:
        Average influence spread (expected number of influenced nodes)
    """
    sources = set(S)
    inf = 0
    for _ in range(R):
        source_set = sources.copy()
        queue = deque(source_set)
        # Forward BFS: propagate from seed nodes
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                # Try to activate children with probability = edge weight
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        inf += len(source_set)
        
    return inf / R  # Average over R simulations

def workerMC(x):
    """
    Worker function for parallel Monte Carlo computation.
    
    Args:
        x: Tuple of (graph, seed_set, num_trials)
        
    Returns:
        Influence spread result
    """
    return computeMC(x[0], x[1], x[2])

def computeRR(graph, S, R, cache=None):
    """
    Compute influence spread using Reverse Reachable (RR) set method.
    
    Algorithm:
    1. Generate R reverse reachable sets by starting from random nodes and
       following reverse edges (backward propagation)
    2. Count how many RR sets are "covered" by seed set S
    3. Influence = (covered_RR_sets / total_RR_sets) * num_nodes
    
    The RR method is efficient because:
    - RR sets can be pre-generated and cached
    - Different seed sets can reuse the same RR sets
    - Only need to check coverage, not re-simulate propagation
    
    Args:
        graph: Graph object
        S: Seed set (list of node indices)
        R: Number of RR sets to generate
        cache: Optional list to store/generate RR sets
        
    Returns:
        Estimated influence spread (expected number of influenced nodes)
    """
    covered = 0
    generate_RR = False
    
    # If cache exists and is non-empty, use cached RR sets (fast path)
    if cache is not None:
        if len(cache) > 0:
            # Check coverage: count RR sets that contain at least one seed node
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True  # Cache is empty, need to generate RR sets

    # Generate R reverse reachable sets
    for i in range(R):
        # Start from a random target node
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        
        # Backward BFS: traverse reverse edges to find all nodes that can reach the target
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                # Check all parent nodes (reverse edges) with probability based on edge weight
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        
        # Check if this RR set is covered by seed set S
        # A set is "covered" if it contains at least one seed node
        for s in S:
            if s in source_set:
                covered += 1
                break
        
        # Store RR set in cache if caching is enabled
        if generate_RR:
            cache.append(source_set)
    
    # Influence = (covered sets / total sets) * total nodes
    return covered * 1.0 / R * graph.num_nodes

def computeSIS(graph, S, beta=0.3, gamma=0.1, num_steps=1):
    status = {node: 'S' for node in range(graph.num_nodes)}
    infected_set = set()
    for i in S:
        i = i.item()
        status[i] = 'I'
        infected_set.add(i)

    for step in range(num_steps):
        new_status = status.copy()
        for node in range(graph.num_nodes):
            if status[node] == 'I':
                neighbors = []
                edges = graph.edge_index.t().tolist()
                for t in edges:
                    if t[0] == node or t[1] == node:
                        # print(t)
                        if t[0] == node:
                            neighbors.append(t[1])
                        else:
                            neighbors.append(t[0])

                for neighbor in neighbors:
                    if status[neighbor] == 'S' and random.random() < beta:
                        new_status[neighbor] = 'I'
                        infected_set.add(neighbor)

                if random.random() < gamma:
                    new_status[node] = 'S'

        status = new_status

    return len(infected_set)

def computeIC(graph, S, steps=1):
    influenced = set(S)
    new_influenced = set(S)
    deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes)

    def get_neighbors(node, edge_index):
        src, dst = edge_index
        neighbors = dst[src == node]
        return neighbors

    for _ in range(steps):
        current_influenced = set()
        for node in new_influenced:
            neighbors = get_neighbors(node, graph.edge_index)
            for neighbor in neighbors:
                if neighbor not in influenced:
                    p = 0.8
                    if random.random() < p:
                        current_influenced.add(neighbor)
        new_influenced = current_influenced
        influenced.update(new_influenced)
        int_set = set()
        for item in influenced:
            if isinstance(item, torch.Tensor):
                int_item = int(item.item())
            else:
                int_item = int(item)

            int_set.add(int_item)

        if not new_influenced:
            break
    return len(int_set)

def computeLT(graph, S, steps=1):
    influenced = set(S)
    new_influenced = set(S)

    for _ in range(steps):
        current_influenced = set()
        for node in new_influenced:
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if neighbor not in influenced:
                    neighbor_influence = sum(1 for n in graph.neighbors(neighbor) if n in influenced) / graph.degree(neighbor)
                    threshold = graph.nodes[neighbor]['threshold']
                    if neighbor_influence >= threshold:
                        current_influenced.add(neighbor)
        new_influenced = current_influenced
        influenced.update(new_influenced)
        if not new_influenced:
            break
    return len(influenced)


def workerRR(x):
    ''' for multiprocessing '''
    return computeRR(x[0], x[1], x[2])

def computeRR_inc(graph, S, R, cache=None, l_c=None):
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True

    for i in range(R):
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        for s in S:
            if s in source_set:
                covered += 1
                break
        if generate_RR:
            cache.append(source_set)
    return covered * 1.0 / R * graph.num_nodes


if __name__ == '__main__':
    # path of the graph file
    path = "../soc-dolphins.txt"
    # number of parallel processes
    num_process = 5
    # number of trials
    num_trial = 10000
    # load the graph
    graph = read_graph(path, ind=1, directed=False)
    print('Generating seed sets:')
    list_S = []
    for _ in range(10):
      list_S.append(random.sample(range(graph.num_nodes), k=random.randint(3, 10)))
      print(f'({str(list_S[-1])[1:-1]})')

    print('Cached single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    RR_cache = []
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial, cache=RR_cache))
      times.append(time.time() - time_start)
    time_2 = time.time()

    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    print('No-cache single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    print('Multi-process MC:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      with Pool(num_process) as p:
        es_infs.append(statistics.mean(p.map(workerMC, [[graph, S, num_trial // num_process] for _ in range(num_process)])))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')
