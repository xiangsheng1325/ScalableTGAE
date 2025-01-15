import sys
import numpy as np
#from re import T
# from turtle import forward
from numpy import eye
import torch
from dgl import function as fn
from collections.abc import Mapping, Iterable
from dgl.nn.functional import edge_softmax
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
from layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
DTYPE = torch.float32
import scipy.sparse as sp

class EmbeddingEncoder(nn.Module):
    def __init__(self, N, H, pretrained_emb=None):
        super(EmbeddingEncoder, self).__init__()
        self.encoding = nn.Embedding(N, H)
        if pretrained_emb is not None:
            print('Using pretrained node embedding!')
            self.encoding.weight.data.copy_(torch.tensor(pretrained_emb))
            self.encoding.weight.requires_grad = True
    def forward(self, pretrained_emb=None):
        if pretrained_emb is not None:
            print('Using pretrained node embedding!')
            self.encoding.weight.data.copy_(pretrained_emb)
            self.encoding.weight.requires_grad = True
        return self.encoding.weight


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
        

class LinearDecoder(nn.Module):
    def __init__(self, N, H):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(H, N)
    def forward(self, X):
        # X: N_all * H
        #    T * N * H
        # Linear: H * N
        # output: N_all * N
        return self.decoder(X)


class NonLinearDecoder(nn.Module):
    def __init__(self, N, H):
        super(NonLinearDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(H, H),
                                     nn.ReLU(),
                                     nn.Linear(H, N))
    def forward(self, X):
        return self.decoder(X)


class G_temporal(nn.Module):
    def __init__(self, N_all, N, T, H, pretrained_emb=None):
        super(G_temporal, self).__init__()
        self.encoder = EmbeddingEncoder(N_all, H, pretrained_emb=pretrained_emb)
        self.decoder = LinearDecoder(N=N, H=H)
        
    def add_loss(self):
        return 0
    
    def forward(self, pretrained_emb=None):
        W = self.decoder(self.encoder(pretrained_emb=pretrained_emb))
        # W -= W.max(dim=-1, keepdims=True)[0]
        return W

class GATLayer(nn.Module):
    def __init__(self, in_dim=128, hid_dim=32, n_heads=4):
        super(GATLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.emb_src = nn.Linear(in_dim, hid_dim * n_heads)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, n_heads, hid_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, n_heads, hid_dim)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.skip_feat = nn.Linear(in_dim, hid_dim * n_heads)
        self.gate = nn.Linear(3 * hid_dim * n_heads, 1)
        self.norm = nn.LayerNorm(hid_dim * n_heads)
        self.activation = nn.PReLU(init=0.25)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, feat):
        feat_src = self.emb_src(feat).view(-1, self.n_heads, self.hid_dim)
        feat_dst = feat_src[:graph.number_of_dst_nodes()]
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft'].reshape(-1, self.hid_dim*self.n_heads)
        skip_feat = self.skip_feat(feat)[:graph.number_of_dst_nodes()]
        gate = torch.sigmoid(self.gate(torch.cat([rst, skip_feat, rst - skip_feat], dim=-1)))
        rst = gate * rst + (1 - gate) * skip_feat
        return self.activation(self.norm(rst))

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim=128, hid_dim=32, n_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.sqrt_dk = self.head_dim ** 0.5
        # 多头线性变换
        self.Q = nn.Linear(in_dim, hid_dim)
        self.K = nn.Linear(in_dim, hid_dim)
        self.V = nn.Linear(in_dim, hid_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, graph, feat):
        # 线性变换得到 Q, K, V
        Q = self.Q(feat).view(-1, self.n_heads, self.head_dim)
        K = self.K(feat).view(-1, self.n_heads, self.head_dim)
        V = self.V(feat).view(-1, self.n_heads, self.head_dim)
        # 存储 Q, K, V 到图的源节点数据
        graph.srcdata['Q'] = Q
        graph.srcdata['K'] = K
        graph.srcdata['V'] = V
        # print("graph srcdata:",graph.srcdata.keys())
        # # 打印 Q, K, V 的信息，用于调试
        # print("Q shape:", Q.shape)
        # print("K shape:", K.shape)
        # print("V shape:", V.shape)
        # print("Q dtype:", Q.dtype)
        # print("K dtype:", K.dtype)
        # print("V dtype:", V.dtype)
        # 检查存储是否成功
        if 'Q' not in graph.srcdata or 'K' not in graph.srcdata or 'V' not in graph.srcdata:
            print("Keys not found in graph.srcdata after update:", graph.srcdata.keys())
            raise KeyError("Keys 'Q' or 'K' or 'V' not found in graph.srcdata")
        # 计算注意力得分
        # graph.srcdata.update({'Q': Q, 'K': K, 'V': V})
        #获取源节点的 K 和目标节点的 Q , 使用 torch.einsum 计算注意力得分
        src_K = graph.srcdata['K']
        dst_Q = graph.dstdata.get('Q', graph.srcdata['Q'][:graph.number_of_dst_nodes()])
        # 确保 src_K 和 dst_Q 的节点数量一致
        if src_K.shape[0] > dst_Q.shape[0]:
            pad_size = src_K.shape[0] - dst_Q.shape[0]
            device = src_K.device  # 获取 src_K 的设备
            dst_Q = torch.cat([dst_Q, torch.zeros(pad_size, self.n_heads, self.head_dim, device=device)], dim=0)
        else:
            pad_size = dst_Q.shape[0] - src_K.shape[0]
            device = dst_Q.device  # 获取 dst_Q 的 device
            src_K = torch.cat([src_K, torch.zeros(pad_size, self.n_heads, self.head_dim, device=device)], dim=0)
        # 使用 torch.einsum 计算注意力得分
        scores = torch.einsum('bhd,bhd->bh', src_K, dst_Q)
        # 确保 scores 的维度与边的数量匹配
        num_edges = graph.number_of_edges()
        if scores.shape[0] < num_edges:
            pad_size = num_edges - scores.shape[0]
            device = scores.device  # 获取 scores 的设备
            padding = torch.zeros(pad_size, scores.shape[1], device=device)
            scores = torch.cat((scores, padding), dim=0)
        elif scores.shape[0] > num_edges:
            scores = scores[:num_edges]
        graph.edata['score'] = scores.unsqueeze(-1)
        # 缩放分数
        graph.edata['score'] = graph.edata['score'] / self.sqrt_dk
        graph.edata['score'] = F.softmax(graph.edata['score'], dim=-1)
        graph.edata['score'] = self.attn_dropout(graph.edata['score'])
        # 消息传递
        graph.update_all(fn.u_mul_e('V', 'score', 'm'), fn.sum('m', 'attn_out'))
        attn_out = graph.dstdata['attn_out'].view(-1, self.hid_dim)
        # 残差连接和层归一化
        # feat = self.norm1(feat + attn_out)
      
        # 调整 attn_out 和 feat 的维度使其匹配
        min_shape = min(attn_out.shape[0], feat.shape[0])
        if feat.is_sparse:
            feat = feat.to_dense()
        attn_out = attn_out[:min_shape]
        feat = feat[:min_shape]
        # 调整 feat 的特征维度使其与 attn_out 的特征维度匹配
        if feat.shape[1]!= attn_out.shape[1]:
            if feat.shape[1] > attn_out.shape[1]:
                feat = feat[:, :attn_out.shape[1]]
            else:
                # 填充 feat 使其维度与 attn_out 匹配
                pad_size = attn_out.shape[1] - feat.shape[1]
                padding = torch.zeros(feat.shape[0], pad_size, device=feat.device)
                feat = torch.cat([feat, padding], dim=1)
        feat = self.norm1(attn_out + feat)
        # 前馈网络
        ffn_out = self.ffn(feat)
        # 残差连接和层归一化
        out = self.norm2(feat + ffn_out)
        return out

class ScalableTGAE(nn.Module):
    def __init__(self, in_dim=128, hid_dim=32, n_heads=4, out_dim=128, dropout=0.1):
        super(ScalableTGAE, self).__init__()
        # self.input_encoder = nn.Linear(out_dim, in_dim)
        # 使用新的 GraphTransformerLayer 作为 attention 层
        self.attention_encoder = GraphTransformerLayer(in_dim=in_dim, hid_dim=hid_dim, n_heads=n_heads, dropout=dropout)
        # self.decoder = nn.Linear(n_heads * hid_dim, out_dim)
        self.decoder = nn.Linear(hid_dim, out_dim)

    def forward(self, blocks, feat):
        # 确保 blocks 中的第一个图块和 feat 输入符合 GraphTransformerLayer 的输入要求
        # blocks 是图块列表，feat 是节点特征
        # 这里将 blocks[0] 作为图输入，feat 作为节点特征输入
        return self.decoder(self.attention_encoder(blocks[0], feat))

# class ScalableTGAE(nn.Module):
#     def __init__(self, in_dim=128, hid_dim=32, n_heads=4, out_dim=128):
#         super(ScalableTGAE, self).__init__()
#         self.attention_encoder = GraphTransformerLayer(in_dim=in_dim, hid_dim=hid_dim, n_heads=n_heads)
#         self.decoder = nn.Linear(n_heads * hid_dim, out_dim)
    
#     def forward(self,adjacency_matrix,feat):
#         encoded_feat = self.attention_encoder(adjacency_matrix,feat)
#         return self.decoder(encoded_feat)

def coo_to_csp(sp_coo):
    num = sp_coo.shape[0]
    feat_num = sp_coo.shape[1]
    row = sp_coo.row
    col = sp_coo.col
    sp_tensor = torch.sparse.FloatTensor(torch.LongTensor(np.stack([row, col])),
                                         torch.tensor(sp_coo.data),
                                         torch.Size([num, feat_num]))
    return sp_tensor
# def coo_to_csp(sp_coo):
#     num = sp_coo.shape[0]
#     feat_num = sp_coo.shape[1]
#     row = sp_coo.row
#     col = sp_coo.col
#     sp_tensor = torch.sparse.FloatTensor(torch.LongTensor(np.stack([row, col])),
#                                          torch.tensor(sp_coo.data),
#                                          torch.Size([num, feat_num]))
#     # 确保返回的是张量
#     return sp_tensor.to_dense()
if __name__ == '__main__':
    import dgl
    import torch
    import numpy as np
    import scipy.sparse as sp
    import os
    from scalable_temporal_graph_autoencoder import FromTemporalGraphToSparseAdj
    from dgl.dataloading import MultiLayerFullNeighborSampler
    from dgl.dataloading import DataLoader
    label_adj, nids = FromTemporalGraphToSparseAdj()
    label_mat = label_adj.tocsr()[nids, :]
    t = 195
    num_nodes = 1899
    feat = sp.diags(np.ones(num_nodes * t).astype(np.float32)).tocsr()
    # feat_tensor = torch.sparse_coo_tensor(torch.tensor([feat.col, feat.row]), feat.data)
    adj = label_adj.tocsr()
    # adj_tensor = torch.sparse_coo_tensor(torch.tensor([src, dst]), adj.data)
    dgl_g = dgl.load_graphs(os.path.join("./data/DBLP/", "dgl_graph.bin"))[0][0]
    # dgl_g = dgl.to_bidirected(dgl_g, copy_ndata=True)
    dgl_g = dgl.add_self_loop(dgl_g)
    train_sampler = MultiLayerFullNeighborSampler(num_layers=1)
    train_dataloader = DataLoader(dgl_g,
                                      nids=torch.from_numpy(nids).long(),
                                      block_sampler=train_sampler,
                                      device='cpu',
                                      batch_size=128,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=0)
    device_id = 'cuda:0'
    model = ScalableTGAE(in_dim=num_nodes * t, hid_dim=32, n_heads=4, out_dim=num_nodes).to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)
    for epoch in range(400):
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            model.train()
            batch_inputs, batch_labels = coo_to_csp(feat[input_nodes, :].tocoo()).to(device_id), \
                                         coo_to_csp(adj[seeds, :].tocoo()).to_dense().to(device_id)
            blocks = [block.to(device_id) for block in blocks]
            train_batch_logits = model(blocks, batch_inputs)
            num_edges = batch_labels.sum() / 2
            loss = -0.5 * torch.sum(batch_labels * torch.log_softmax(train_batch_logits, dim=-1)) / num_edges
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 10 == 0:
                print("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}".format(epoch+1, step+1, loss.cpu().data))
            else:
                sys.stdout.flush()
                sys.stdout.write("Epoch: {:03d}, Step: {:03d}, loss: {:.7f}\r".format(epoch+1, step+1, loss.cpu().data))
                sys.stdout.flush()
