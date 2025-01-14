import datetime
import os.path
import pickle
from curses import raw
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader
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
from eval_utils import compute_graph_statistics

def FromTemporalGraphToSparseAdj(filename='./data/DBLP/edgelist_new.txt',
                                 save_path='./data/DBLP/dgl_graph.bin'):
    
    DBLP = np.loadtxt(filename, dtype=int)
    TemporalGraph = {
        "src": DBLP[:, 0],
        "dst": DBLP[:, 1],
        "timestamp": DBLP[:, 2]
    }
    node_num = len(np.unique(np.concatenate([TemporalGraph["src"], TemporalGraph["dst"]])))
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
    dgl.data.utils.save_graphs(save_path, [dglg])
    target_src = TemporalGraph["src"] + node_num * TemporalGraph["timestamp"]
    target_dst = TemporalGraph["dst"]
    train_nids = np.unique(target_src)
    #打印 target_nids
    print(f"target_nids: {train_nids.max()}")
    print(f'TemporalGraph: {TemporalGraph["src"]}')
    # 打印 target_src 和 target_dst 的基本信息
    print(f"target_src 最大值: {target_src.max()}, 最小值: {target_src.min()}, 唯一值数量: {len(np.unique(target_src))}")
    print(f"target_dst 最大值: {target_dst.max()}, 最小值: {target_dst.min()}, 唯一值数量: {len(np.unique(target_dst))}")

    # 打印时间戳范围
    print(f"time_unique 最大值: {time_unique.max()}, 最小值: {time_unique.min()}, 唯一值数量: {len(np.unique(time_unique))}")

    # 打印矩阵形状
    shape=(node_num * (time_unique.max() - time_unique.min() + 1), node_num)
    print(f"矩阵 shape: {shape}")
    return sp.coo_matrix((np.ones(len(target_src)), (target_src, target_dst)),
                         shape=(node_num * (time_unique.max() - time_unique.min() + 1), node_num)), train_nids

DEVICE = "cuda:0"
EDGE_OVERLAP_LIMIT = {
    'CORA-ML' : 0.9,
}
MAX_STEPS = 400

def random_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def compute_temporal_graph_statistics(A_T):
    seq_len = A_T.shape[0]//A_T.shape[1]
    num_nodes = A_T.shape[1]
    A_seq = [A_T[i*num_nodes:(i+1)*num_nodes, :] for i in range(seq_len)]
    stats = []
    for i in range(seq_len):
        stats.append(compute_graph_statistics(A_seq[i]))
        
    return stats

def cal_avg_median_stats(adj, gen_mat):
    stats_real = compute_temporal_graph_statistics(adj)
    stats_gen_mat = compute_temporal_graph_statistics(gen_mat)
    
    f_avg = {}
    f_med = {}
    for key in stats_real[0].keys():
        stats_real_values = np.array([stat[key] for stat in stats_real])
        stats_gen_mat_values = np.array([stat[key] for stat in stats_gen_mat])
        if 0 in stats_real_values:
            #vals = abs((stats_real_values - stats_gen_mat_values))
            pass
        else:
            vals = abs((stats_real_values - stats_gen_mat_values) / stats_real_values) 
            #vals = abs((stats_real_values - stats_gen_mat_values))
            f_avg[key] = np.mean(vals)
            
            vals = vals[vals != 0]
            f_med[key] = np.median(vals)
    return f_avg, f_med
    
class ParseArguments(object):
    def __init__(self):
        self.device = 'cuda:0'
        self.n_layers = 2  # 增加层数
        self.H = 256       # 增加隐藏层维度
        self.n_heads = 8   # 增加注意力头数
        self.batch_size = 64  # 减小批量大小
        self.g_type = 'temporal'
        self.lr = 1e-3     # 降低学习率
        self.weight_decay = 5e-4  # 增加权重衰减
        self.max_epochs = 300     # 减少最大迭代次数
        self.graphic_mode = 'overlay'  # 改变模式
        self.criterion = 'eo'
        self.eo_limit = 0.95  # 调整限制
        self.seed = 2024   
           
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
    best_results = {}
    sp.save_npz(os.path.join("./data/DBLP", "adj.npz"), adj)
    
    dgl_g = dgl.load_graphs(os.path.join("./data/DBLP", "dgl_graph.bin"))[0][0]
    dgl_g = dgl.add_self_loop(dgl_g)
    train_sampler = MultiLayerFullNeighborSampler(num_layers=args.n_layers)
    train_dataloader = DataLoader(dgl_g.to(args.device),
                                      indices=torch.from_numpy(nids).long().to(args.device),
                                      graph_sampler=train_sampler,
                                      device=args.device,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=0)
    model = ScalableTGAE(in_dim=num_nodes * t,
                         hid_dim=int(args.H/args.n_heads),
                         n_heads=args.n_heads,
                         out_dim=num_nodes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    
    best_f_avg = {}
    best_f_med = {}
   
    for epoch in range(args.max_epochs):
        num_edges_all = 0
        num_loss_all = 0
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
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
            if (step+1) % 50 == 0:
                print("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}".format(epoch+1, step+1, loss.cpu().data))
            else:
                sys.stdout.flush()
                sys.stdout.write("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}\r".format(epoch+1, step+1, loss.cpu().data))
                sys.stdout.flush()
        scheduler.step()
        print("Epoch: {:03d}, overall loss: {:.7f}".format(epoch + 1, num_loss_all/num_edges_all))
        if (epoch+1) % 50 == 0:
            gen_mat = sp.csr_matrix(adj.shape)
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                    test_inputs = coo_to_csp(feat[input_nodes.cpu(), :].tocoo()).to(args.device)
                    blocks = [block for block in blocks]
                    test_batch_logits = torch.softmax(model(blocks, test_inputs), dim=-1)
                    num_edges = adj[seeds.cpu(), :].sum()
                    gen_mat[seeds.cpu(), :] = edge_from_scores(test_batch_logits.cpu().numpy(), num_edges)
                    if (step+1) % 20 == 0:
                        print("Epoch: {:03d}, Generating Step: {:03d}".format(epoch+1, step+1))
                    else:
                        sys.stdout.flush()
                        sys.stdout.write("Epoch: {:03d}, Generating Step: {:03d}\r".format(epoch+1, step+1))
                        sys.stdout.flush()
            eo = adj.multiply(gen_mat).sum() / adj.sum()
            sp.save_npz(os.path.join("./data/DBLP", "gen_mat.npz".format(epoch+1)), gen_mat)
            print("Epoch: {:03d}, Edge Overlap: {:07f}".format(epoch + 1, eo))
            
            print('\n\n', '='*80, sep='')
            f_avg, f_med= cal_avg_median_stats(adj, gen_mat)
            for key in f_avg.keys():
                print("f_avg({}): {:.6f}".format(key, f_avg[key]))
                print("f_med({}): {:.6f}".format(key, f_med[key]))
                if key not in best_f_avg or f_avg[key] < best_f_avg[key]:
                    best_f_avg[key] = f_avg[key]
                if key not in best_f_med or f_med[key] < best_f_med[key]:
                    best_f_med[key] = f_med[key]
            print('='*80, '\n\n', sep='')
    print("Finished training.")
    for key in best_f_avg.keys():
        print("Best f_avg({}): {:.6f}".format(key, best_f_avg[key]))
        print("Best f_med({}): {:.6f}".format(key, best_f_med[key]))
            