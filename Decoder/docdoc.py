import networkx as nx
import matplotlib.pyplot as plt
import random
import torch
import pickle
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def rgb(rgb):
    return tuple([x/255.0 for x in rgb])

def normalize_adj(mx):
    """Row-normalize sparse matrix（行规范化稀疏矩阵）"""
    rowsum = np.array(mx.sum(1))  # 每行相加的和
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # 生成对角矩阵
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)  # transpose函数相当于换索引位置，类似于转置


# 加载数据集
def load_dataset(name):
    if name == 'Jazz':
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
        return to_networkx(data, to_undirected=True)
    elif name=='cora':
        with open('data/cora/cora_ml_mean_IC10.SG', 'rb') as f:
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
        return to_networkx(data, to_undirected=True)
    else:
        with open('data/network/netscience.SG', 'rb') as f:
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
        return to_networkx(data, to_undirected=True)


# Independent Cascade Model (IC) 模拟传播过程
def independent_cascade(graph, seeds, p=0.1):
    new_active, A = seeds[:], seeds[:]
    while new_active:
        new_ones = []
        for node in new_active:
            neighbors = set(graph.neighbors(node)) - set(A)
            for neighbor in neighbors:
                if random.random() <= p:
                    new_ones.append(neighbor)
        new_active = list(set(new_ones))
        A += new_active
    return A


# Linear Threshold (LT) 模型传播过程
def linear_threshold(graph, seeds, thresholds=None):
    if thresholds is None:
        thresholds = {node: random.random() for node in graph.nodes()}

    activated = set(seeds)
    new_active = set(seeds)

    while new_active:
        next_active = set()
        for node in graph.nodes():
            if node not in activated:
                influence = sum(1 for neighbor in graph.neighbors(node) if neighbor in activated)
                if influence / graph.degree(node) >= thresholds[node]:
                    next_active.add(node)

        new_active = next_active - activated
        activated.update(new_active)

    return activated


# 可视化传播过程
def visualize_graph(graph, seeds, infected, pos, title):
    node_colors = []
    for node in graph.nodes():
        if node in seeds:
            node_colors.append('red')  # 种子节点
        elif node in infected:
            node_colors.append('orange')  # 被感染节点
        else:
            node_colors.append('green')  # 正常节点

    plt.figure(figsize=(8, 8))
    nx.draw(graph, pos, node_color=node_colors, with_labels=False, node_size=50, font_size=8, edge_color='gray')

    # 添加图例
    legend_labels = {
        'Seed Nodes': 'red',
        'Infected Nodes': 'orange',
        'Normal Nodes': 'green'
    }
    for label, color in legend_labels.items():
        plt.scatter([], [], color=color, label=label)
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='upper right')

    plt.title(title)
    plt.show()

    plt.savefig('dataset:{title}', format='pdf')
    plt.close()



# 主函数
def main():
    datasets = ['Jazz']
    seed_fraction = 0.1
    p = 0.05  # IC模型的传播概率

    for dataset_name in datasets:
        G = load_dataset(dataset_name)

        # 移除自环
        G.remove_edges_from(nx.selfloop_edges(G))

        num_nodes = G.number_of_nodes()
        num_seeds = int(seed_fraction * num_nodes)
        seeds = random.sample(list(G.nodes()), num_seeds)
        print(seeds)

        # 保持一致的布局
        pos = nx.spring_layout(G, seed=42)  # 固定布局位置

        # IC模型传播
        infected_nodes_ic = independent_cascade(G, seeds, p)
        print(infected_nodes_ic)
        visualize_graph(G, seeds, infected_nodes_ic, pos, f'{dataset_name} - IC Model')

        infected_nodes_ic = independent_cascade(G, seeds, p)
        visualize_graph(G, seeds, infected_nodes_ic, pos, f'{dataset_name} - IC Model')

        # LT模型传播
        infected_nodes_lt = linear_threshold(G, seeds)
        visualize_graph(G, seeds, infected_nodes_lt, pos, f'{dataset_name} - LT Model')

        infected_nodes_lt = linear_threshold(G, seeds)
        visualize_graph(G, seeds, infected_nodes_lt, pos, f'{dataset_name} - LT Model')


if __name__ == "__main__":
    main()

