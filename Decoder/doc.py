import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import random

# 读取JAZZ数据集，并创建一个无向图
# G = nx.read_edgelist('path_to_jazz_dataset.txt', nodetype=int)  # 替换为实际的路径


def normalize_adj(mx):
    """Row-normalize sparse matrix（行规范化稀疏矩阵）"""
    rowsum = np.array(mx.sum(1))  # 每行相加的和
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # 生成对角矩阵
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)  # transpose函数相当于换索引位置，类似于转置


# file_path = 'data/fb_food/fb-pages-food.edges'
# dataset1 = Planetoid(root='data/Planetoid', name=dataset)
# data = dataset1[0]
with open('data/jazz/jazz.SG', 'rb') as f:
    graph = pickle.load(f)  # 将f中的数据序列化

# with open('data/citeseer/citeseer.SG', 'rb') as f:
#     graph = pickle.load(f)

adj = graph['adj']
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse()
# 转换为 PyTorch 张量
# adj = torch.tensor(adj, dtype=torch.float)
edge_index = adj.indices()
edge_attr = adj.values()
# 获取节点数
num_nodes = adj.shape[0]
# 构建 PyG 数据对象，不包含节点特征
data = Data(edge_index=edge_index, num_nodes=num_nodes)

# 转换为 NetworkX 图
G = to_networkx(data, to_undirected=True)

total_nodes = len(G.nodes())

# 定义每个模型的种子节点集
#0.1(0.54):106, 17, 74, 77, 110, 128, 6, 71, 12, 168, 60, 59, 144, 153, 2, 82, 73, 122, 181
#0.2(0.67):190, 89, 186, 125, 55, 44, 189, 94, 184, 138, 5, 182, 168, 18, 183, 117, 26, 85, 129, 106, 49, 2, 96, 29, 73, 43, 11, 118, 87, 119, 4, 121, 145, 63, 68, 6, 112, 179, 98
seed_nodes_ic = {190, 89, 186, 125, 55, 44, 189, 94, 184, 138, 5, 182, 168, 18, 183, 117, 26, 85, 129, 106, 49, 2, 96, 29, 73, 43, 11, 118, 87, 119, 4, 121, 145, 63, 68, 6, 112, 179, 98}  # IC模型的种子节点（示例）
seed_nodes_lt = {2, 7, 14}  # LT模型的种子节点（示例）
#0.1(0.12):131, 135, 134, 167, 121, 173, 163, 104, 95, 193, 82, 98, 191, 157, 68, 107, 69, 130, 100
#0.2(0.72):100, 31, 167, 130, 131, 169, 94, 178, 69, 110, 170, 109, 52, 191, 107, 95, 134, 97, 157, 59, 80, 195, 4, 98, 194, 117, 53, 193, 153, 163, 6, 104, 121, 68, 113, 122, 135, 166, 173
seed_nodes_deepim = {131, 135, 134, 167, 121, 173, 163, 104, 95, 193, 82, 98, 191, 157, 68, 107, 69, 130, 100}  # DeepIM模型的种子节点（示例）
#0.05(0.74):134, 173, 121, 99, 104, 59, 68, 148, 6, 135
#0.1(0.85):134, 173, 121, 99, 104, 59, 68, 148, 6, 135, 191, 131, 167, 107, 82, 95, 163, 98, 100, 69
#0.2(0.91):98, 167, 193, 31, 104, 69, 110, 170, 148, 94, 53, 100, 194, 178, 163, 97, 4, 191, 195, 80, 113, 121, 52, 107, 169, 134, 131, 99, 153, 173, 59, 122, 34, 130, 152, 48, 135, 117, 68, 82
seed_nodes_tapim = {134, 173, 121, 99, 104, 59, 68, 148, 6, 135, 191, 131, 167, 107, 82, 95, 163, 98, 100, 69}  # TapIM模型的种子节点（示例）

# 定义每个模型的感染节点比例5467  # IC模型感染节点比例 20%
infected_percentage_ic = 0.54
infected_percentage_lt = 0.3  # LT模型感染节点比例 30%
infected_percentage_deepim = 0.12  # DeepIM模型感染节点比例 25%
infected_percentage_tapim = 0.80  # TapIM模型感染节点比例 35%

# 随机选择感染节点
def select_infected_nodes(G, seed_nodes, infected_percentage):
    num_infected = int(infected_percentage * total_nodes)
    # possible_nodes = set(G.nodes()) - seed_nodes
    possible_nodes = set(G.nodes())
    infected_nodes = set(random.sample(possible_nodes, num_infected))
    return infected_nodes

# 获取每个模型的感染节点集
infected_nodes_ic = select_infected_nodes(G, seed_nodes_ic, infected_percentage_ic)
infected_nodes_lt = select_infected_nodes(G, seed_nodes_lt, infected_percentage_lt)
infected_nodes_deepim = select_infected_nodes(G, seed_nodes_deepim, infected_percentage_deepim)
infected_nodes_tapim = select_infected_nodes(G, seed_nodes_tapim, infected_percentage_tapim)
#
r=[39,71,83]
r = [h / 255.0 for h in r]

o=[138,176,124]
o = [i / 255.0 for i in o]

g=[238,238,185]
g = [e / 255.0 for e in g]

# 绘制函数
def draw_graph(G, seed_nodes, infected_nodes, title):
    # 设置节点颜色
    color_map = []
    # 移除自环
    G.remove_edges_from(nx.selfloop_edges(G))
    for node in G:
        if node in seed_nodes:
            color_map.append(r)  # 种子节点为红色242,45,47
        elif node in infected_nodes:
            color_map.append(o)  # 感染节点为橙色205,107,39
        else:
            color_map.append(g)  # 正常节点为绿色117,148,100

    # 使用 spring_layout 布局来将图形呈现为圆形 seed_positions=True
    pos = nx.spring_layout(G, seed=42)
    # pos = nx.spring_layout(G, seed=42)

    # 绘制图形
    plt.figure(figsize=(9, 9))
    nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=180, font_size=10, edge_color='gray')
    # nx.draw(G, node_color=color_map, with_labels=False, node_size=20, font_size=0, edge_color='gray')
    plt.title(title)
    plt.show()

# # 绘制IC模型的传播结果
# draw_graph(G, seed_nodes_ic, infected_nodes_ic, "IC Model Spread")
#
# # 绘制LT模型的传播结果
# # draw_graph(G, seed_nodes_lt, infected_nodes_lt, "LT Model Spread")
#
# # 绘制DeepIM模型的传播结果
# draw_graph(G, seed_nodes_deepim, infected_nodes_deepim, "DeepIM Model Spread")

# 绘制TapIM模型的传播结果
draw_graph(G, seed_nodes_tapim, infected_nodes_tapim, "TapIM Model Spread")
