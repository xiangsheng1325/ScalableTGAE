from data_utils import *
from eval_utils import *
from model_utils import *
from models import *
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.sparse as sp

def loadGameGraph(game, N, binary=True):
	# N is the number of players in the game
	# load csv data
	# unweighted and weighted networks can be loaded exactly the same way
	# below shows the loader for weighted networks
	if binary:
		df_network = pd.read_csv(f'{src}/network{game}.csv', index_col = 0)
	else:
		df_network = pd.read_csv(f'{src}/network{game}_weighted.csv', index_col = 0)


	# T is number of timestamps (10 frames)
	T = len(df_network)
	# load VFOA network to T x N x (N+1) array
	# vfoa[t, n, i] is the probability of player n+1 looking at object i at time t
	# i: 0 - laptop, 1 - player 1, 2 - player 2, ..., N - player N
	vfoa = np.reshape(df_network.values, (T,N,N+1))

	# print information
	print(f'network id:{game}\t length(x 1/3 second): {T}\t num of players: {N}')
	return vfoa

src = './data' # root dir of data
meta = pd.read_csv('./data/network_list.csv')
gdata = []
for _, row in meta.iterrows():
	gdata.append(loadGameGraph(row['NETWORK'], row['NUMBER_OF_PARTICIPANTS']))
	break
# T x N x (N+1)
myg = gdata[0]
# 把看laptop变换到对角线对应值
newg = np.stack([x[:, 1:]+np.diag(x[:, 0]) for x in myg])

DEVICE = "cuda:0"
EDGE_OVERLAP_LIMIT = {
    'CORA-ML' : 0.7, 
    'Citeseer' : 0.8,
    'PolBlogs': 0.41,
    'RT-GOP': 0.7
}
# MAX_STEPS = 400
MAX_STEPS = 50


def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class parse_arguments(object):
    def __init__(self):
        self.data_path = "./game_network_0.npy"
        self.data_name = 'game_0'
        self.statistics_step = 10
        self.number_of_samples = 5
        self.H = 9
        self.g_type = 'temporal'
        self.lr = 0.05
        self.weight_decay = 1e-6
        self.graphic_mode = 'overlap'
        self.fig_path = None
        #self.fig_path = 'game_network.pdf'
        self.table_path = 'logs/game.csv'
        self.eo_limit = 1.0
        self.criterion = 'eo'
        self.seed = 42


def main(args):
    # 导入邻接矩阵
    _A_obs, _X_obs, _z_obs = load_npy(args.data_path)

    # 将输入图变为无向、对称阵
    temp_A_obs = _A_obs.toarray().reshape(-1, _A_obs.shape[1], _A_obs.shape[1])
    for i in range(len(temp_A_obs)):
        temp_A_obs[i] = temp_A_obs[i] + temp_A_obs[i].T + np.eye(temp_A_obs[i].shape[0])
    _A_obs = sp.csr_matrix(temp_A_obs.reshape(_A_obs.shape[0], _A_obs.shape[1]))

    #_A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    #_A_obs = _A_obs - sp.eye(_A_obs.shape[0], _A_obs.shape[0])
    _A_obs[_A_obs < 0] = 0
    #lcc = largest_connected_components(_A_obs)
    #_A_obs = _A_obs[lcc,:][:,lcc]
    _N = _A_obs.shape[0]

    # 验证边的比例
    val_share = 0.05
    # 测试边的比例
    test_share = 0.1
    seed = 42

    # 将邻接矩阵分割为训练、验证、测试集
    train_ones, val_ones, val_zeros, test_ones, test_zeros = train_val_test_split_adjacency(_A_obs, 
                                                                                            val_share, 
                                                                                            test_share, 
                                                                                            seed,
                                                                                            every_node=False,
                                                                                            undirected=False, 
                                                                                            connected=False, 
                                                                                            asserts=False)
    # 训练集转为稀疏矩阵
    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    # assert (train_graph.toarray() == train_graph.toarray().T).all()

    # 边重叠限制
    if args.eo_limit is not None:
        edge_overlap_limit = args.eo_limit 
    elif args.data_name in EDGE_OVERLAP_LIMIT.keys():
        edge_overlap_limit = EDGE_OVERLAP_LIMIT[args.data_name]
    else:
        edge_overlap_limit = 0.6

    training_stat = dict()
    
    # 图类型
    if args.g_type != 'all':
        training_stat[args.g_type] = list()
    else:
        training_stat = {i: list() for i in ['cell', 'fc', 'svd']}
    
    df = pd.DataFrame()
    # 对所有模型
    for model_name in list(training_stat.keys()):
        print(f"'{model_name.upper()}' Approach")
        # 优化器参数
        optimizer_args = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if model_name == 'fc':
            optimizer_args['weight_decay'] = 1e-4
        if args.criterion == 'eo':
            # 边重叠标准，调用invoke进行检查，当输入两图的边重叠度（multiply）达到阈值则停止训练
            callbacks = [EdgeOverlapCriterion(invoke_every=5, 
                                            edge_overlap_limit=edge_overlap_limit)]
        elif args.criterion == 'val':
            # 连接预测标准
            callbacks = [LinkPredictionCriterion(invoke_every=2,
                                                val_ones=val_ones,
                                                val_zeros=val_zeros,
                                                max_patience=5)]

        if args.fig_path is not None:
            invoke_every = args.statistics_step 
        else:
            invoke_every = MAX_STEPS + 1
        stat_collector = StatisticCollector(invoke_every, _A_obs, test_ones, test_zeros, graphic_mode=args.graphic_mode, n_samples=args.number_of_samples)
        callbacks.append(stat_collector)
        
        print("newg.shape[0]: ", newg.shape[0])
        print("_A_obs.shape[0]: ", _A_obs.shape[0])

        # 模型定义
        model = Cell(A=train_graph,
                    H=args.H,
                    T=newg.shape[0],
                    g_type=model_name,
                    callbacks=callbacks,
                    device=DEVICE)

        # 模型训练
        model.train(steps=MAX_STEPS,
                optimizer_fn=torch.optim.Adam,
                optimizer_args=optimizer_args)

        # 计算参数
        stat_collector.invoke(None, model)
        training_stat[model_name] = stat_collector.training_stat
        stats = training_stat[model_name][-1]['stats']
        stat_df = pd.DataFrame({k: [s[k] for s in stats] for k in stats[0].keys()})
        stat_df = stat_df.mean()
        df[model_name] = stat_df.T

    #original_stat = compute_graph_statistics(_A_obs)
    #df[args.data_name] = list(original_stat.values()) + [1, 1, 1]
    if args.table_path is not None:
        df.to_csv(args.table_path)

    if args.fig_path is not None:
        fig, axs = plt.subplots(3, 3, figsize=(15,9))
        fig.suptitle(args.data_name, fontsize=18)
        for stat_id, (stat_name, stat) in enumerate(
            list(zip(['Max.Degree','Assortativity', 'Power law exp.', 'Rel. edge distr. entr', 
            'Clustering coeff.', 'Gini coeff.','Wedge count', 'Triangle count', 'Square count'], 
            ['d_max', 'assortativity', 'power_law_exp', 'rel_edge_distr_entropy', 'clustering_coefficient', 'gini',
            'wedge_count', 'triangle_count', 'square_count']))):

            axs[stat_id // 3, stat_id % 3].set_ylabel(stat_name, fontsize=18)
            if args.graphic_mode == 'overlap':
                axs[stat_id // 3, stat_id % 3].set_xlabel('Edge overlap (in %)', fontsize=13)
            else:
                axs[stat_id // 3, stat_id % 3].set_xlabel('Iterations', fontsize=13)
            axs[stat_id // 3, stat_id % 3].axhline(y=original_stat[stat], color='g', linestyle='--', label='target')
            for model_name, model_statistic in training_stat.items():
                if args.graphic_mode == 'overlap':
                    xs = [100 * i['overlap'] for i in model_statistic]
                else:
                    xs = [i['iteration'] for i in model_statistic]
                    
                axs[stat_id // 3, stat_id % 3].errorbar(xs, 
                                                        [np.mean([j[stat] for j in i['stats']]) for i in model_statistic],
                                                        [np.std([j[stat] for j in i['stats']]) for i in model_statistic],
                                                        ls='none',
                                                        fmt='.',
                                                        label=model_name
                                                        )

        axLine, axLabel = axs[stat_id // 3, stat_id % 3].get_legend_handles_labels()
        fig.legend(axLine, axLabel, loc = 'center right', fontsize=15)
        fig.tight_layout()
        if args.fig_path is not None:
            plt.savefig(args.fig_path)
        plt.close()


args = parse_arguments()
if args.seed is not None:
    random_seed(args.seed)
main(args)