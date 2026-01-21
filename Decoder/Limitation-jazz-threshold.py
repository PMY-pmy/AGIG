import networkx as nx
import random
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

def normalize_adj(mx):
    """Row-normalize sparse matrix（行规范化稀疏矩阵）"""
    rowsum = np.array(mx.sum(1))  # 每行相加的和
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # 生成对角矩阵
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)  # transpose函数相当于换索引位置，类似于转置

# Step 1: Load the JAZZ dataset
# Here, we use a synthetic example; replace this with actual loading of the JAZZ dataset
with open('data/jazz/jazz.SG', 'rb') as f:
    graph = pickle.load(f)  # 将f中的数据序列化

adj = graph['adj']
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse()

edge_index = adj.indices()
edge_attr = adj.values()
# 获取节点数
num_nodes = adj.shape[0]
# 构建 PyG 数据对象，不包含节点特征
data = Data(edge_index=edge_index, num_nodes=num_nodes)
# 转换为 NetworkX 图
G = to_networkx(data, to_undirected=True)
# G = from_networkx(G)
# num_nodes = G.num_nodes
# G = nx.karate_club_graph()  # Placeholder, replace with JAZZ dataset

# Step 2: Randomly select 30% of nodes as seed nodes
num_nodes = G.number_of_nodes()
seed_count = int(0.3 * num_nodes)
seed_nodes = random.sample(G.nodes(), seed_count)


# Step 3: Define a function to simulate Independent Cascade (IC) model
def independent_cascade(G, seed_nodes, threshold):
    activated_nodes = set(seed_nodes)
    new_activated_nodes = set(seed_nodes)

    while new_activated_nodes:
        new_activated = set()
        for node in new_activated_nodes:
            neighbors = set(G.neighbors(node)) - activated_nodes
            for neighbor in neighbors:
                if random.random() < threshold:
                    new_activated.add(neighbor)
        new_activated_nodes = new_activated
        activated_nodes.update(new_activated_nodes)

    return float(len(activated_nodes)/num_nodes)


# Step 4: Repeat for IC, LT, and SIS models with different thresholds
thresholds = [0.05, 0.16, 0.4]
ic_results = []

for threshold in thresholds:
    spread = independent_cascade(G, seed_nodes, threshold)
    ic_results.append(spread)

print("IC model spreads:", ic_results)


# Step 5: Define functions for Linear Threshold (LT) and SIS models
def linear_threshold(G, seed_nodes, threshold1):
    activated_nodes = set(seed_nodes)
    node_thresholds = {n: random.uniform(0, 1) for n in G.nodes()}

    new_activated_nodes = set(seed_nodes)
    while new_activated_nodes:
        new_activated = set()
        for node in set(G.nodes()) - activated_nodes:
            neighbors = set(G.neighbors(node)) & activated_nodes
            influence = sum(1 for _ in neighbors) / G.degree(node)
            if influence > threshold1:
                new_activated.add(node)
        new_activated_nodes = new_activated
        activated_nodes.update(new_activated_nodes)

    return float(len(activated_nodes)/num_nodes)


def sis_model(G, seed_nodes, infection_prob=0.005, recovery_prob=0.003, max_iter=100):
    infected = set(seed_nodes)
    susceptible = set(G.nodes()) - infected
    for _ in range(max_iter):
        new_infected = set()
        for node in infected:
            for neighbor in G.neighbors(node):
                if neighbor in susceptible and random.random() < infection_prob:
                    new_infected.add(neighbor)
        recovered = set(node for node in infected if random.random() < recovery_prob)
        infected.update(new_infected)
        infected -= recovered
        susceptible -= new_infected
    return float(len(infected)/num_nodes)


lt_results = []
sis_results = []

for threshold1 in [0.34, 0.46, 0.76]:
    spread_lt = linear_threshold(G, seed_nodes, threshold1)
    lt_results.append(spread_lt)

spread_sis = sis_model(G, seed_nodes)
sis_results.append(spread_sis)

spread_sis = sis_model(G, seed_nodes, 0.001, 0.001)
sis_results.append(spread_sis)

spread_sis = sis_model(G, seed_nodes, 0.006, 0.001)
sis_results.append(spread_sis)

print("LT model spreads:", lt_results)
print("SIS model spreads:", sis_results)
