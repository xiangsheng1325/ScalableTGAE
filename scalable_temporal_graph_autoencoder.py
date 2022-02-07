import datetime
import os.path
import pickle
from curses import raw
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
import networkx as nx
import dgl
import numpy as np
from data_utils import *
from eval_utils import *
from model_utils import *
from models import *
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.sparse as sp


def loadGraph(filename):
    arr = np.load(filename)
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    return sp.csr_matrix(arr)


def LoadTemporalGraph(filename="/home/xuchenhao/datasets/CollegeMsg/CollegeMsg.txt",
                      save_path=None):
    """
    读temporal graph文件名，输出temporal graph
    :param filename: 数据集文件名，默认为edgelist，header默认为None，每一行默认为(src, dst, unix时间戳)
    :return: 保存了source node，target node， 时间戳的字典
    """
    temporal_dict = dict()
    temporal_dict["src"] = []
    temporal_dict["dst"] = []
    temporal_dict["timestamp"] = []
    f = open(filename)
    tmp_data = f.readlines()
    f.close()
    for line in tmp_data:
        line = line.strip().split(" ")
        temporal_dict["src"].append(int(line[0]))
        temporal_dict["dst"].append(int(line[1]))
        tmp_time = time.localtime(int(line[2]))
        temporal_dict["timestamp"].append(datetime.date(tmp_time[0], tmp_time[1], tmp_time[2]))
    time_ids = pd.Series(temporal_dict.get("timestamp"))
    time_ids = (time_ids - time_ids.min()).apply(lambda x: x.days)
    temporal_dict["timestamp"] = time_ids.values
    node_set = set(temporal_dict["src"]+temporal_dict["dst"])
    node_name_to_idx = {node_name: i for i, node_name in enumerate(list(node_set))}
    src_names = pd.Series(temporal_dict.get("src"))
    dst_names = pd.Series(temporal_dict.get("dst"))
    src_names = src_names.apply(lambda x: node_name_to_idx[x])
    dst_names = dst_names.apply(lambda x: node_name_to_idx[x])
    temporal_dict["src"] = src_names.values
    temporal_dict["dst"] = dst_names.values
    if save_path is None:
        save_path = os.path.join("/home/xuchenhao/datasets/CollegeMsg/", "edgelist_date.csv")
    else:
        save_path = os.path.join(save_path, "edgelist_date.csv")
    pd.DataFrame(temporal_dict).to_csv(save_path, sep=" ", header=False, index=False)
    temporal_dict["node_to_idx"] = node_name_to_idx
    return temporal_dict


def FromTemporalGraphToSparseAdj(filename="/home/xuchenhao/datasets/CollegeMsg/CollegeMsg.txt",
                                 save_path=None):
    if save_path is None:
        save_path = "/home/xuchenhao/datasets/CollegeMsg/"
    TemporalGraph = LoadTemporalGraph(filename=filename, save_path=save_path)
    node_num = len(TemporalGraph["node_to_idx"])
    time_unique = np.unique(TemporalGraph["timestamp"])
    temporal_src = TemporalGraph["src"] + node_num * TemporalGraph["timestamp"]
    temporal_dst = TemporalGraph["dst"] + node_num * TemporalGraph["timestamp"]
    inner_src = np.array(range(time_unique.min() * node_num, time_unique.max() * node_num))
    inner_dst = np.array(range((time_unique.min() + 1) * node_num, (time_unique.max() + 1) * node_num))
    self_src = np.array(range(time_unique.min() * node_num, (time_unique.max() + 1) * node_num))
    self_dst = np.array(range(time_unique.min() * node_num, (time_unique.max() + 1) * node_num))
    temporal_edges = (np.concatenate([temporal_src, inner_src, self_src]),
                      np.concatenate([temporal_dst, inner_dst, self_dst]))
    dglg = dgl.graph(temporal_edges)
    save_path = os.path.join("/home/xuchenhao/datasets/CollegeMsg/", "dgl_graph.bin")
    dgl.data.utils.save_graphs(save_path, [dglg])
    target_src = TemporalGraph["src"] + node_num * TemporalGraph["timestamp"]
    target_dst = TemporalGraph["dst"]
    train_nids = np.unique(target_src)
    return sp.coo_matrix((np.ones(len(target_src)), (target_src, target_dst)),
                         shape=(node_num * (time_unique.max() - time_unique.min() + 1), node_num)), train_nids



DEVICE = "cuda:0"
EDGE_OVERLAP_LIMIT = {
    'CORA-ML' : 0.9,
}
MAX_STEPS = 400
# MAX_STEPS = 120


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def param_compare(s1, s2, o):
    d1 = pd.read_csv(s1).drop(columns=["Unnamed: 0"])
    d2 = pd.read_csv(s2).drop(columns=["Unnamed: 0"])
    d3 = abs(d1 - d2) / d1
    s1 = d3.mean()
    s2 = d3.median()
    s1.name = 'mean'
    s2.name = 'median'
    d3 = d3.append(s1)
    d3 = d3.append(s2)
    d3.to_csv(o, index=False)


class parse_arguments(object):
    def __init__(self):
        self.data_path = "/home/xuchenhao/graph_generation/experiment/TGAE/data/DBLP/DBLP_evolve.npy"
        self.data_name = 'DBLP'
        self.directed = False       # 是否有向
        self.self_loop = True
        self.evolve = True
        self.statistics_step = 10
        self.number_of_samples = 5
        self.H = 128
        self.g_type = 'temporal'
        self.lr = 0.05
        self.weight_decay = 1e-6
        self.graphic_mode = 'overlap'
        self.criterion = 'eo'
        self.eo_limit = 0.99
        self.seed = 42
        self.fig_path = None
        #self.fig_path = 'game_network.pdf'
        self.raw_param_path = '/home/xuchenhao/graph_generation/experiment/TGAE/data/DBLP/DBLP_raw_param_undir_selfloop.csv'
        # self.raw_param_path = './data/DBLP/DBLP_raw_param_undir_selfloop.csv'
        self.table_path = './tables/{}_steps={}_eo={}_lr={}_H={}.csv'.format(self.data_name, MAX_STEPS, self.eo_limit, self.lr, self.H)
        self.graph_path = './graphs/{}_steps={}_eo={}_lr={}_H={}.npy'.format(self.data_name, MAX_STEPS, self.eo_limit, self.lr, self.H)   # path to save generated graphs
        self.result_path = './results/{}_steps={}_eo={}_lr={}_H={}.csv'.format(self.data_name, MAX_STEPS, self.eo_limit, self.lr, self.H)


def main(args):
    # 预训练Embedding
    # look_back = 2
    # pretrained_emb = np.load("/home/xuchenhao/datasets/DBLP/node_embedding/DBLP_dynAERNN_evolve_embs.npy")
    # pretrained_emb = np.array([pretrained_emb[0 if i < look_back else i - look_back] for i in range(pretrained_emb.shape[0] + look_back)])
    # pretrained_emb = pretrained_emb.reshape((-1, pretrained_emb.shape[2]))

    # _A_obs, _X_obs, _z_obs = load_npy(args.data_path)
    _A_obs = loadGraph(args.data_path)

    N = _A_obs.shape[1]
    T = _A_obs.shape[0] // _A_obs.shape[1]
    temp_A_obs = _A_obs.toarray().reshape(-1, _A_obs.shape[1], _A_obs.shape[1])
    # if not args.directed:
    #     for i in range(len(temp_A_obs)):
    #         temp_A_obs[i] = temp_A_obs[i] + temp_A_obs[i].T
    #         temp_A_obs[temp_A_obs > 1] = 1
    if args.self_loop:
        for i in range(T):
            temp_A_obs[i] = temp_A_obs[i] - np.eye(temp_A_obs[i].shape[0])
            temp_A_obs[temp_A_obs < 0] = 0

    # if args.self_loop:
    #     # 保存对角线全0的邻接矩阵
    #     temp_A_obs_t = copy.deepcopy(temp_A_obs)
    #     for i in range(len(temp_A_obs)):
    #         temp_A_obs[i] = temp_A_obs[i] + np.eye(temp_A_obs[i].shape[0])
    #         temp_A_obs_t[i] = temp_A_obs_t[i] - np.eye(temp_A_obs[i].shape[0])
    #     _A_obs_t = sp.csr_matrix(temp_A_obs_t.reshape(_A_obs.shape[0], _A_obs.shape[1]))
    #     _A_obs_t[_A_obs_t > 1] = 1
    #     _A_obs_t[_A_obs_t < 0] = 0
    _A_obs = sp.csr_matrix(temp_A_obs.reshape(_A_obs.shape[0], _A_obs.shape[1]))

    # _A_obs[_A_obs > 1] = 1
    # _A_obs[_A_obs < 0] = 0
    # lcc = largest_connected_components(_A_obs)
    # _A_obs = _A_obs[lcc,:][:,lcc]
    # _N = _A_obs.shape[0]

    # 求边数量
    # if args.self_loop:
    #     edge_num = [np.count_nonzero(_A_obs_t[t * N : (t + 1) * N].toarray()) // 2 for t in range(T)]
    # else:
    if not args.directed:
        edge_num = [np.count_nonzero(_A_obs[t * N : (t + 1) * N].toarray()) // 2 for t in range(T)]
    else:
        edge_num = [np.count_nonzero(_A_obs[t * N : (t + 1) * N].toarray()) for t in range(T)]

    # raw_g = _A_obs.toarray().reshape(-1, _A_obs.shape[1], _A_obs.shape[1])
    # for i in range(1, len(raw_g)):
    #     raw_g[i] += raw_g[i-1]
    #     raw_g[raw_g > 1] = 1
    # raw_stat = [compute_graph_statistics(sp.csr_matrix(g)) for g in raw_g]
    # raw_df = pd.DataFrame({k: [s[k]  for s in raw_stat] for k in raw_stat[0].keys()})
    # raw_df.to_csv('/home/xuchenhao/graph_generation/experiment/TGAE/data/DBLP/DBLP_raw_param_undir_selfloop.csv')

    # 求mask矩阵
    # if args.self_loop:
    #     nonzero_rows = ~((_A_obs_t.toarray() == 0).all(1))
    # else:
    nonzero_rows = ~((_A_obs.toarray() == 0).all(1))
    mask_mat = np.ones(_A_obs.shape).astype(int) & [[i] for i in nonzero_rows]
    for i in range(T):
        mask_mat[i * N : (i + 1) * N] = mask_mat[i * N : (i + 1) * N] & nonzero_rows[i * N : (i + 1) * N]
    mask_mat = ~(mask_mat.astype(bool))
    # _A_obs = _A_obs[nonzero_rows]

    val_share = 0.1
    test_share = 0.05
    seed = 42

    train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(_A_obs, 
                                                                                            val_share, 
                                                                                            test_share, 
                                                                                            seed,
                                                                                            every_node=False,
                                                                                            undirected=False, 
                                                                                            connected=False, 
                                                                                            asserts=False)
    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1])), shape=_A_obs.shape).tocsr()
    # assert (train_graph.toarray() == train_graph.toarray().T).all()
    # print("val_ones: ", val_ones)

    if args.eo_limit is not None:
        edge_overlap_limit = args.eo_limit 
    elif args.data_name in EDGE_OVERLAP_LIMIT.keys():
        edge_overlap_limit = EDGE_OVERLAP_LIMIT[args.data_name]
    else:
        edge_overlap_limit = 0.6

    training_stat = dict()
    
    if args.g_type != 'all':
        training_stat[args.g_type] = list()
    else:
        training_stat = {i: list() for i in ['cell', 'fc', 'svd']}
    
    df = pd.DataFrame()
    for model_name in list(training_stat.keys()):    
        print(f"'{model_name.upper()}' Approach")
        optimizer_args = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if model_name == 'fc':
            optimizer_args['weight_decay'] = 1e-4
        if args.criterion == 'eo':
            callbacks = [EdgeOverlapCriterion(invoke_every=5, 
                                            edge_overlap_limit=edge_overlap_limit)]
        elif args.criterion == 'val':
            callbacks = [LinkPredictionCriterion(invoke_every=2,
                                                val_ones=val_ones,
                                                val_zeros=val_zeros,
                                                max_patience=5)]

        if args.fig_path is not None:
            invoke_every = args.statistics_step 
        else:
            invoke_every = MAX_STEPS + 1
        stat_collector = StatisticCollector(invoke_every, _A_obs, test_ones, test_zeros, graphic_mode=args.graphic_mode, n_samples=args.number_of_samples, evolve=args.evolve, g_path=args.graph_path)
        callbacks.append(stat_collector)

        model = Cell(A=train_graph, 
                    H=args.H, 
                    T=_A_obs.shape[0] // _A_obs.shape[1],
                    g_type=model_name, 
                    callbacks=callbacks,
                    device=DEVICE,
                    pretrained_emb=None,
                    mask_mat=mask_mat,
                    edge_num=edge_num,
                    directed=args.directed,
                    self_loop=args.self_loop,)

        model.train(steps=MAX_STEPS,
                optimizer_fn=torch.optim.Adam,
                optimizer_args=optimizer_args)

        stat_collector.invoke(None, model, nonzero_rows, last=True, self_loop=args.self_loop)
        training_stat[model_name] = stat_collector.training_stat
        stats = training_stat[model_name][-1]['stats']
        stat_df = pd.DataFrame({k: np.mean([[t[k] for t in s] for s in stats], axis=0) for k in stats[0][0].keys()})
        # stat_df = stat_df.mean()
        # df[model_name] = stat_df.T

    # original_stat = compute_graph_statistics(_A_obs)
    # df[args.data_name] = list(original_stat.values()) + [1, 1, 1]
    if args.table_path is not None:
        stat_df.to_csv(args.table_path, index=False)
        if args.raw_param_path and args.result_path:
            param_compare(args.raw_param_path, args.table_path, args.result_path)


# args = parse_arguments()
# if args.seed is not None:
#     random_seed(args.seed)
# main(args)


class ParseArguments(object):
    def __init__(self):
        self.data_path = "/home/xuchenhao/datasets/CollegeMsg/CollegeMsg.txt"
        self.data_name = 'CollegeMsg'
        self.device = 'cuda:0'
        self.statistics_step = 10
        self.number_of_samples = 5
        self.n_layers = 1
        self.H = 128
        self.n_heads = 4
        self.batch_size = 128
        self.g_type = 'temporal'
        self.lr = 0.0003
        self.weight_decay = 1e-6
        self.max_epochs = 200
        self.graphic_mode = 'overlap'
        self.criterion = 'eo'
        self.eo_limit = 0.99
        self.seed = 42


args = ParseArguments()
if args.seed is not None:
    random_seed(args.seed)


if __name__ == '__main__':
    label_adj, nids = FromTemporalGraphToSparseAdj()
    label_mat = label_adj.tocsr()[nids, :]
    num_nodes = label_adj.shape[1]
    t = label_adj.shape[0] // num_nodes
    feat = sp.diags(np.ones(num_nodes * t).astype(np.float32)).tocsr()
    adj = label_adj.tocsr()
    dgl_g = dgl.load_graphs(os.path.join("/home/xuchenhao/datasets/CollegeMsg/", "dgl_graph.bin"))[0][0]
    dgl_g = dgl.add_self_loop(dgl_g)
    train_sampler = MultiLayerFullNeighborSampler(n_layers=args.n_layers)
    train_dataloader = NodeDataLoader(dgl_g.to(args.device),
                                      nids=torch.from_numpy(nids).long().to(args.device),
                                      block_sampler=train_sampler,
                                      device=args.device,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=0)
    model = ScalableTGAE(in_dim=num_nodes * t,
                         hid_dim=int(args.H/args.n_heads),
                         n_heads=args.n_heads,
                         out_dim=num_nodes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.max_epochs):
        num_edges_all = 0
        num_loss_all = 0
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            model.train()
            batch_inputs, batch_labels = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args.device), \
                                         coo_to_csp(adj[seeds.cpu(), :].tocoo()).to_dense().to(args.device)
            blocks = [block.to(args.device) for block in blocks]
            train_batch_logits = model(blocks, batch_inputs)
            num_edges = batch_labels.sum() / 2
            num_edges_all += num_edges
            loss = -0.5 * torch.sum(batch_labels * torch.log_softmax(train_batch_logits, dim=-1)) / num_edges
            num_loss_all += loss.cpu().data * num_edges
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 20 == 0:
                print("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}".format(epoch+1, step+1, loss.cpu().data))
            else:
                sys.stdout.flush()
                sys.stdout.write("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}\r".format(epoch+1, step+1, loss.cpu().data))
                sys.stdout.flush()
        print("Epoch: {:03d}, overall loss: {:.7f}".format(epoch + 1, num_loss_all/num_edges_all))