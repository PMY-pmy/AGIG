import numpy as np
import torch
# import wandb
from torch_geometric.data import Data
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, SGD
import pickle as pkl
import time
import matplotlib.pyplot as plt
import argparse, json
import pickle


import scipy.sparse as sp

from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse.linalg

import argparse
import random
import utils.graph_utils as graph_utils
import torch.nn.functional as F

from Decoder.dt.envs import environment
from Encoder.GT.models import GraphTransformerNet
from Encoder.main import MyGT_gen, MyGT_train

from dt.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from dt.models.DGT import DecisionTransformer, GraphTransformerNet
from dt.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cpu')
    # log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    #第一组大规模数据
    # data_list = []
    # # for f in ['reed','amherst','cornell','jh','Carnegie49','Princeton12','WashU32','Yale4']:
    # dataset_list = ['Carnegie49']
    # for f in dataset_list:
    #     DATAPATH = '/home/pmy/MyDT2/data/LinkThief-main/data/fb_{}'.format(f)
    #     fp = os.path.join(DATAPATH, 'adj.pkl')
    #     with open(fp, 'rb') as f:
    #         dataA = pkl.load(f)
    #         edges = dataA.nonzero()
    #         edges1 = torch.tensor(edges)
    #     fp1 = os.path.join(DATAPATH, 'features1.pkl')
    #     with open(fp1, 'rb') as f:
    #         dataFea1 = pkl.load(f)
    #     edge_attr = torch.zeros(edges1.size(1), dtype=torch.float)
    #     for i in range(edges1.size(1)):
    #         node1 = edges1[0, i]
    #         node2 = edges1[1, i]
    #         edge_attr[i] = torch.norm(torch.from_numpy(dataFea1[node1]) - torch.from_numpy(dataFea1[node2]))
    #     data = Data(x=dataFea1, edge_index=edges1, edge_attr=edge_attr)
    #     data_list.append(data)

   #第二组对比数据
    def normalize_adj(mx):
        """Row-normalize sparse matrix（行规范化稀疏矩阵）"""
        rowsum = np.array(mx.sum(1))  # 每行相加的和
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # 生成对角矩阵
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)  # transpose函数相当于换索引位置，类似于转置

    with open('/home/pmy/DeepIM/data/cora_ml_mean_IC10.SG', 'rb') as f:
        graph = pickle.load(f)

    adj, inverse_pairs = graph['adj'], graph['inverse_pairs']
    num_node = adj.shape[0]
    dataFea = torch.zeros([num_node,1], dtype=torch.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.Tensor(adj.toarray()).to_sparse()
    index = adj.indices()
    e = adj.values()
    num_node = adj.shape[0]
    data = []
    data1 = Data(x = dataFea, edge_index=index, edge_attr=e)
    data.append(data1)

   #第三组简单图数据
    if os.path.isdir(args.graph):
        path_graphs = [os.path.join(args.graph, file_g) for file_g in os.listdir(args.graph) if
                       not file_g.startswith('.')]
    else:  # read one graph
        path_graphs = [args.graph]
    # graph_lst = [graph_utils.read_graph(path_g, ind=0, directed=False) for path_g in path_graphs]
    graph_lst = [graph_utils.read_graph(path_g, ind=0, directed=True) for path_g in path_graphs]
    for i in range(len(path_graphs)):
        graph_lst[i].path_graph = path_graphs[i]

    args.graphs = graph_lst
    # args.graphs = data_list
    args.graphs = data
    if env_name == 'IM':
        # train_env
        env = environment.Environment(env_name, data, budget=10, method='SIS', use_cache=True)
        # max_ep_len = 0.1 * env.graphs.num_nodes #选择出的action的个数
        env_targets = [900]  # evaluation conditioning targets,用于评估算法在环境中的表现

        # scale = 1000.  # normalization for rewards/returns，对奖励或回报值的规范化
    else:
        raise NotImplementedError
    #原本括号里面是dataset_list
    for i in range(len(data)):
        data_train = []
        # data_train.append(data_list[i])
        data_train.append(data[i])
        env = environment.Environment(env_name, data_train, budget=10, method='SIS', use_cache=True)
        env.reset(idx=0)
        state = env.state
        # state = np.array(state)
        state = torch.tensor(state)
        graph = env.graph
        state_dim = graph.num_nodes
        act_dim = 1
        fea = graph.x
        # max_ep_len = int(0.1 * env.graph.num_nodes)
        max_ep_len = 20

        K = variant['K']
        batch_size = variant['batch_size']
        num_eval_episodes = variant['num_eval_episodes']

        # pct_traj = variant.get('pct_traj', 1.)

        # 以上都是在生成一个可以处理的数据输入
        def eval_episodes(target_rew):
            def fn(model):
                returns, lengths = [], []
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        if model_type == 'dt':
                            ret, length, act_s, cur_states = evaluate_episode_rtg(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                # scale=scale,
                                # target_return=target_rew/scale,
                                target_return=target_rew,
                                mode='normal',
                                # state_mean=state_mean,
                                # state_std=state_std,
                                device=device,
                            )
                        else:
                            ret, length = evaluate_episode(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                # target_return=target_rew/scale,
                                target_return=target_rew,
                                # mode=mode,
                                # state_mean=state_mean,
                                # state_std=state_std,
                                device=device,
                            )
                    returns.append(ret)
                    lengths.append(length)
                return {
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                    f'actions': act_s,
                    f'current_states': cur_states,
                }

            return fn

        if model_type == 'dt':
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=max_ep_len,  # K是token的个数
                max_ep_len=max_ep_len,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4 * variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
            )
        else:
            raise NotImplementedError

        model = model.to(device=device)

        warmup_steps = variant['warmup_steps']
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        optimizer_my = torch.optim.SGD(
            model.parameters(),
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1)
        )

        with open("/home/pmy/Encoder/configs/configs1.json") as f:
            config = json.load(f)

        params = config['params']
        # if args.seed is not None:
        #     params['seed'] = int(args.seed)
        # if args.epochs is not None:
        #     params['epochs'] = int(args.epochs)
        # if args.batch_size is not None:
        #     params['batch_size'] = int(args.batch_size)
        # if args.init_lr is not None:
        #     params['init_lr'] = float(args.init_lr)
        # if args.lr_reduce_factor is not None:
        #     params['lr_reduce_factor'] = float(args.lr_reduce_factor)
        # if args.lr_schedule_patience is not None:
        #     params['lr_schedule_patience'] = int(args.lr_schedule_patience)
        # if args.min_lr is not None:
        #     params['min_lr'] = float(args.min_lr)
        # if args.weight_decay is not None:
        #     params['weight_decay'] = float(args.weight_decay)
        # if args.print_epoch_interval is not None:
        #     params['print_epoch_interval'] = int(args.print_epoch_interval)
        # if args.max_time is not None:
        #     params['max_time'] = float(args.max_time)
        # network parameters
        net_params = {}
        net_params_r = {}
        net_params = config['net_params']
        net_params['device'] = 'cuda:0'
        net_params['batch_size'] = params['batch_size']
        # if args.L is not None:
        #     net_params['L'] = int(args.L)
        # if args.hidden_dim is not None:
        #     net_params['hidden_dim'] = int(args.hidden_dim)
        # if args.out_dim is not None:
        #     net_params['out_dim'] = int(args.out_dim)
        # if args.residual is not None:
        #     net_params['residual'] = True if args.residual == 'True' else False
        # if args.edge_feat is not None:
        #     net_params['edge_feat'] = True if args.edge_feat == 'True' else False
        # if args.readout is not None:
        #     net_params['readout'] = args.readout
        # if args.n_heads is not None:
        #     net_params['n_heads'] = int(args.n_heads)
        # if args.in_feat_dropout is not None:
        #     net_params['in_feat_dropout'] = float(args.in_feat_dropout)
        # if args.dropout is not None:
        #     net_params['dropout'] = float(args.dropout)
        # if args.layer_norm is not None:
        #     net_params['layer_norm'] = True if args.layer_norm == 'True' else False
        # if args.batch_norm is not None:
        #     net_params['batch_norm'] = True if args.batch_norm == 'True' else False
        # if args.self_loop is not None:
        #     net_params['self_loop'] = True if args.self_loop == 'True' else False
        # if args.lap_pos_enc is not None:
        #     net_params['lap_pos_enc'] = True if args.pos_enc == 'True' else False
        # if args.pos_enc_dim is not None:
        #     net_params['pos_enc_dim'] = int(args.pos_enc_dim)
        # if args.wl_pos_enc is not None:
        #     net_params['wl_pos_enc'] = True if args.pos_enc == 'True' else False

        # graph1
        net_params['node_in_dim'] = graph.x.shape[1]
        net_params['edge_in_dim'] = 1

        # root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        #     config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        # root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        #     config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        # write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        #     config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        # write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        #     config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        # dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

        # if not os.path.exists(out_dir + 'results'):
        #     os.makedirs(out_dir + 'results')
        #
        # if not os.path.exists(out_dir + 'configs'):
        #     os.makedirs(out_dir + 'configs')
        net_params_r = net_params
        if model_type == 'dt':
            model1 = GraphTransformerNet(
                net_params_r
            )
        else:
            raise NotImplementedError
        model1 = model1.to(device=device)

        optimizer1 = optim.Adam(model1.parameters(), lr=1e-2, weight_decay=1e-2)
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min',
                                                         factor=0.5,
                                                         patience=15,
                                                         verbose=True)

        if model_type == 'dt':
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                # get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )
        else:
            raise NotImplementedError

        def laplacian_positional_encoding(g, pos_enc_dim):
            # edge_index = g.num_edges
            # head = []
            # end = []
            weight = []
            device = torch.device('cuda:0')
            if g.edge_index.device != device:
                g.edge_index = g.edge_index.to(device)
            num_nodes = g.num_nodes
            # for key, value in g.edges.items():
            #     print(f'key: {key}, Vlue: {value}')
            #     key1, key2 = key
            #     head.append(key1)
            #     end.append(key2)
            #     weight.append(value)
            edge_index = torch.tensor(g.edge_index, dtype=torch.long)
            g.edge_index = edge_index
            # g.features = torch.tensor(g.features, dtype=torch.float)
            g.features = g.features
            edge_weight = torch.tensor(g.edge_attr, dtype=torch.float)
            edge_weight = edge_weight.view(len(edge_weight), 1)
            if edge_weight.device != device:
                edge_weight = edge_weight.to(device)
            g.edge_weight = edge_weight
            if g.edge_weight.device != device:
                g.edge_weight = g.edge_weight.to(device)
            if g.edge_attr.device != device:
                g.edge_attr = g.edge_attr.to(device)

            # 计算拉普拉斯矩阵，并转换为Scipy稀疏矩阵格式
            # edge_weight = torch.ones((int(edge_index),), dtype=torch.float)
            laplacian = get_laplacian(edge_index, edge_weight=edge_weight, normalization='sym', num_nodes=num_nodes)
            laplacian_matrix = to_scipy_sparse_matrix(laplacian[0], edge_attr=laplacian[1], num_nodes=num_nodes)

            # 使用稀疏特征值分解计算特征值和特征向量
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix.asfptype(), k=pos_enc_dim + 1, which='SM')

            # 忽略最小的特征值（通常接近0），并保留其对应的特征向量
            eigvecs = eigvecs[:, 1:pos_enc_dim + 1]  # 去掉第一个特征向量

            # 将特征向量转换为Tensor并作为节点特征
            g.lap_pos_enc = torch.from_numpy(eigvecs).float()

            return g
        input_dim = 1
        output_dim = 1
        input_dim1 = 64
        output_dim1 = 1

        loss_fn = lambda s_hat, s: torch.mean((s_hat - s) ** 2)

        # criterion = nn.CrossEntropyLoss()
        def loss_inverse(y, y_true):
            forward_loss = F.mse_loss(y, y_true)
            # shape[]函数表示矩阵某一维度的长度
            return forward_loss

        def loss_inverse1(y1, y2):
            forward_loss1 = F.mse_loss(y1, y2)
            return forward_loss1

        criterion_my = nn.BCELoss()
        class Trainer:

            def __init__(self, model, model1, optimizer, optimizer1, batch_size, scheduler=None, scheduler1=None, eval_fns=None):
                self.model = model
                self.model1 = model1
                self.optimizer = optimizer
                self.optimizer1 = optimizer1
                self.batch_size = batch_size
                # self.get_batch = get_batch
                # self.loss_fn = loss_fn
                self.scheduler = scheduler
                self.scheduler1 = scheduler1
                self.eval_fns = [] if eval_fns is None else eval_fns
                self.diagnostics = dict()

                self.start_time = time.time()

            def forward(self, states, state_dim, actions, rewards, returns_to_go, timesteps, **kwargs):
                # we don't care about the past rewards in this model
                for i in range(10):
                    states = states.reshape(1, -1, self.model.state_dim)
                    actions = actions.reshape(1, -1, self.model.act_dim)
                    returns_to_go = returns_to_go.reshape(1, -1, 1)
                    timesteps = timesteps.reshape(1, -1)

                    if self.model.max_length is not None:
                        states = states[:, -self.model.max_length:]
                        actions = actions[:, -self.model.max_length:]
                        returns_to_go = returns_to_go[:, -self.model.max_length:]
                        timesteps = timesteps[:, -self.model.max_length:]

                        # pad all tokens to sequence length
                        attention_mask = torch.cat(
                            [torch.zeros(self.model.max_length - states.shape[1]), torch.ones(states.shape[1])])
                        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
                        states = torch.cat(
                            [torch.zeros((states.shape[0], self.model.max_length - states.shape[1], self.model.state_dim),
                                         device=states.device), states],
                            dim=1).to(dtype=torch.float32)
                        actions = torch.cat(
                            [torch.zeros((actions.shape[0], self.model.max_length - actions.shape[1], self.model.act_dim),
                                         device=actions.device), actions],
                            dim=1).to(dtype=torch.float32)
                        returns_to_go = torch.cat(
                            [torch.zeros((returns_to_go.shape[0], self.model.max_length - returns_to_go.shape[1], 1),
                                         device=returns_to_go.device), returns_to_go],
                            dim=1).to(dtype=torch.float32)
                        timesteps = torch.cat(
                            [torch.zeros((timesteps.shape[0], self.model.max_length - timesteps.shape[1]),
                                         device=timesteps.device), timesteps],
                            dim=1
                        ).to(dtype=torch.long)
                    else:
                        attention_mask = None
                    # optimizer.zero_grad()
                    # optimizer1.zero_grad()
                    device = torch.device('cuda:0')
                    action_list =[]
                    state_list = []
                    for i in range(20):
                        self.model.train()
                        self.model1.train()
                        self.optimizer.zero_grad()
                        state_pre = self.model.forward(
                            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
                    # return_preds = torch.tensor(return_preds, dtype=torch.float32, requires_grad=True)
                    # print(action_preds, return_preds)
                    # output_action = action_preds[0, -1]
                    # # 通过一个全连接层将输出的一维action映射成40维（节点数量）
                    # logits = self.model.fc(output_action)
                    # # 再通过softmax函数得到每个节点对应的概率值（这些概率值都在0-1之间，并且和为1）
                    # probs = F.softmax(logits, dim=-1)
                    # # 将概率分布中的最大值以及对应的下标值输出为max_prob/max_index
                    # # max_prob, max_index = torch.randint(probs, dim=-1)
                    # max_index = torch.multinomial(probs, 1).item()
                    # max_prob = probs[max_index].item()
                    #
                    # l = len(rewards)
                    # a = []
                    # for i in range(2, l + 1):
                    #     a.append(actions[:, -i])
                    # for h in a:
                    #     h1 = h.item()
                    #     h1 = int(h1)
                    #     probs[h1] = 0.0
                    # if max_index in actions:
                    #     # 通过索引掩码获取一个不包含指定下标元素的新张量
                    #     # mask = torch.ones(probs.shape, dtype=torch.bool)
                    #     # mask[max_index] = False
                    #     # probs[max_index] = 0.0
                    #     # 使用掩码进行过滤
                    #     # new_probs = probs[mask]
                    #     max_prob1, max_index1 = torch.max(probs, dim=-1)
                    #     return max_index1
                    # # 返回的下标值即为要选出的节点，即action的值(max_index)
                    #以上是原来的，以下是后续修改的
                    # for i in action_list:
                    #     state[int(i)] = 1
                        state = state_pre
                        print(state)
                        # int_action.append(outputs1)
                        # actions[-1] = outputs1
                        # print(f"输出选择出来的actions列表：{int_action}")
                        # 把current_state作为新的一维特征值加入到原有特征矩阵中去
                        graph1 = env.graph
                        # edge_index = graph.edge_index
                        #对比数据时，这些数据集本身是不带节点特征的
                        old_features = graph1.x
                        if isinstance(old_features, np.ndarray):
                            old_features = torch.from_numpy(old_features)
                        # write in singapore
                        # 实验的graph中节点没有特征值，所以我们将输出的state值作为节点的新特征输入[0，1，1，0...]
                        # action = actions
                        # action_L.append(actions)

                        features = state
                        # new_features = features[0]
                        new_features = features.T
                        device = torch.device('cuda:0')
                        if old_features.device != device:
                            old_features = old_features.to(device)
                        if new_features.device != device:
                            new_features = new_features.to(device)
                        graph1.features = torch.cat((old_features, new_features), dim=1)
                        # graph1.features = new_features
                        # graph1.x = graph1.features[:, 1]
                        graph1.x = graph1.features
                        graph1.x = graph1.x.view(2810,1)
                        if net_params_r['lap_pos_enc']:
                            print("[!] Adding Laplacian positional encoding.")
                            graph1 = laplacian_positional_encoding(graph1, net_params_r['pos_enc_dim'])
                        lap_pos_enc = graph1.lap_pos_enc
                        sign_flip = torch.rand(lap_pos_enc.size(1))
                        sign_flip[sign_flip >= 0.5] = 1.0
                        sign_flip[sign_flip < 0.5] = -1.0
                        lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(0)
                        # graph1.x = graph1.x / graph1.x.sum(1, keepdim=True)
                        self.optimizer1.zero_grad()
                        h, e, edge_index = self.model1.forward(graph1, lap_pos_enc)


                        # graph_input_h, graph_input_e, edge_index = MyGT_gen(graph)
                        y = torch.where(h > 0, 1, 0)
                        num_ones = (y == 1).sum().item()
                        influence = num_ones / len(y)
                        print(f'Number of elements equal to 1: {num_ones}')
                        print(f'百分比为: {influence}')
                        y = y.float()
                        y = torch.tensor(y, requires_grad=True)

                        y_true = np.full(h.shape, 1.0)
                        y_true = torch.tensor(y_true, dtype=torch.float32, requires_grad=True,
                                              device=torch.device('cuda:0'))
                        loss = F.mse_loss(h, y_true)
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
                        # self.optimizer.step()
                        self.optimizer1.step()

                        self.optimizer.step()
                        LOSS.append(loss)
                        sorted_tensor, sorted_indices = torch.sort(state_pre, descending=True)
                        node_num1 = sorted_indices.shape[1]
                        node_num_20 = int(node_num1 * 0.2)
                        node_num_10 = int(node_num1 * 0.1)
                        node_num_5 = int(node_num1 * 0.05)
                        node_num_1 = int(node_num1 * 0.01)
                        # 提取前 20% 个元素
                        top20_elements = sorted_tensor[:, :node_num_20]
                        top10_elements = sorted_tensor[:, :node_num_10]
                        top5_elements = sorted_tensor[:, :node_num_5]
                        top1_elements = sorted_tensor[:, :node_num_1]
                        # 提取前 20% 个元素的索引
                        top20_indices = sorted_indices[:, :node_num_20]
                        top10_indices = sorted_indices[:, :node_num_10]
                        top5_indices = sorted_indices[:, :node_num_5]
                        top1_indices = sorted_indices[:, :node_num_1]
                        state_list.append(top20_indices)
                        if influence >= 0.6:
                            top20_indices = top20_indices.view(node_num_20)
                            for i in top20_indices:
                                graph1.features[i, 1] = 1
                            graph1.features[:, 1][graph1.features[:, 1] != 1] = 0
                            print(graph1.features)
                            graph1.x = graph1.features[:, 1]
                            graph1.x =graph1.x.view(node_num1,1)
                            h, e, edge_index = self.model1.forward(graph1, lap_pos_enc)
                            y = torch.where(h > 0, 1, 0)
                            num_ones = (y == 1).sum().item()
                            influence = num_ones / len(y)
                            print(f'Number of elements equal to 1: {num_ones}')
                            print(f'百分比为: {influence}')
                return LOSS, state_list

        model_train = Trainer(model, model1, optimizer, optimizer1, batch_size, scheduler=None, scheduler1=None, eval_fns=None)
        # model_train = model_train.to(device=device)

        # return action_preds, return_preds

        class LinearTransform(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(LinearTransform, self).__init__()
                self.linear = nn.Linear(input_dim, output_dim)

            def forward(self, x):
                return self.linear(x)

        class LinearTransform1(nn.Module):
            def __init__(self, input_dim1, output_dim1):
                super(LinearTransform1, self).__init__()
                self.linear = nn.Linear(input_dim1, output_dim1)

            def forward(self, x):
                return self.linear(x)


        model.train()
        model1.train()
        LOSS = []
        action_L = []
        for iter in range(variant['max_iters']):
            influenced_nodes_list = []
            # states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            ep_return = env_targets
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

            device = torch.device('cuda:0')
            if state.device != device:
                state = state.to(device)
            if actions.device != device:
                actions = actions.to(device)
            if rewards.device != device:
                rewards = rewards.to(device)
            if target_return.device != device:
                target_return = target_return.to(device)
            if timesteps.device != device:
                timesteps = timesteps.to(device)
            int_action = []
            train_start = time.time()
            model_train.model.train()
            for i in range(max_ep_len):
                actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                #以下的输出为LOSS值的列表
                outputs1 = model_train.forward(state.to(dtype=torch.float32),
                                               state_dim,
                                               actions.to(dtype=torch.float32),
                                               rewards.to(dtype=torch.float32),
                                               target_return.to(dtype=torch.float32),
                                               timesteps.to(dtype=torch.long))
                state[int(outputs1)] = 1
                int_action.append(outputs1)
                actions[-1] = outputs1
            # print(f"输出选择出来的actions列表：{int_action}")
            # 把current_state作为新的一维特征值加入到原有特征矩阵中去
            graph = env.graph
            # edge_index = graph.edge_index
            old_features1 = graph.x
            if isinstance(old_features1, np.ndarray):
                old_features = torch.from_numpy(old_features1)
            # write in singapore
            # 实验的graph中节点没有特征值，所以我们将输出的state值作为节点的新特征输入[0，1，1，0...]
            action = actions
            action_L.append(actions)
            print(f"输出选择出来的actions列表：{action}")
            features = state
            # new_features = features[0]
            new_features = features.unsqueeze(1)
            device = torch.device('cuda:0')
            if old_features.device != device:
                old_features = old_features.to(device)

            if new_features.device != device:
                new_features = new_features.to(device)
            graph.features = torch.cat((old_features, new_features), dim=1)
            graph.x = graph.features
            x_hat = graph.features
            x_hat = torch.tensor(x_hat, requires_grad=True)
            # print(graph)

            # optimizer1.zero_grad()
            graph_input_h, graph_input_e, edge_index, lossGT = MyGT_gen(graph)
            # new_params = {'params': optimizer1, 'lr': 1e-2}
            # optimizer.add_param_group(new_params)
            # optimizer_my.add_param_group(new_params)# output h,e
            model_fc = LinearTransform(input_dim, output_dim).to(device)
            model_fc1 = LinearTransform(input_dim1, output_dim1).to(device)
            # graph_input_h = model_fc(graph_input_h)
            # graph_input_e = model_fc1(graph_input_e)
            graph_input = Data(x=graph_input_h, edge_index=edge_index, edge_sttr=graph_input_e)
            # print(graph_input_h, graph_input_e)
            # print(graph_input)
            y = torch.where(graph_input_h > 0, 1, 0)
            num_ones = (y == 1).sum().item()
            influence = num_ones / len(y)
            print(f'Number of elements equal to 1: {num_ones}')
            print(f'百分比为: {influence}')
            y = y.float()
            y = torch.tensor(y, requires_grad=True)
            y_true = torch.ones(y.shape).to(device)
            y_true = y_true.float()
            y_true = torch.tensor(y_true, requires_grad=True)

            loss, L, L_total = loss_inverse(y_true, y, x_hat)
            loss1 = criterion_my(torch.tensor(num_ones / len(y), dtype=torch.float32, requires_grad=True),
                                 torch.tensor(1, dtype=torch.float32, requires_grad=True))
            loss2 = abs(
                torch.tensor(influence, dtype=torch.float32, requires_grad=True) - torch.tensor(1, dtype=torch.float32,
                                                                                                requires_grad=True))
            LOSS.append(loss2.cpu().detach().numpy())
            optimizer_train.zero_grad()
            loss2.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
            optimizer_train.step()
            # optimizer1.step()

            print(f"Epoch {iter}, Loss: {loss2.item()}")
            print(f"Epoch {iter} parameter statistics:")
            # print("Current optimizer parameters:")
            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(f"Parameter group {i}:")
            #     for param in param_group['params']:
            #         if param.requires_grad:
            #             print(param.data)
        print(LOSS, action_L)
        plt.plot(LOSS, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

    # outputs = trainer.train_iteration(state_dim, act_dim, num_steps=variant['num_steps_per_iter'],
    #                                   iter_num=iter + 1, print_logs=True)
    # print(outputs)

    # 通过GT生成新的节点特征表达之后，通过某种手段将所有节点特征聚合为最后的整图表达y值，以表示传播的范围

    # if log_to_wandb:
    #     wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='IM')
    parser.add_argument('--influenced_target_node_num', type=float, default=0.60)
    parser.add_argument('--seed_num_pro', type=float, default=0.10)
    parser.add_argument('--dataset', type=str, default='graph1')
    parser.add_argument('--graph', type=str, metavar='GRAPH_PATH', default='train_data',
                        help='path to the graph file')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2)
    parser.add_argument('--warmup_steps', type=int, default=10000)  # 这个参数是干什么用的？
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=4)
    parser.add_argument('--num_steps_per_iter', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()

    experiment('graph-experiment', variant=vars(args))
