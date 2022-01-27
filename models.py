from re import T
from turtle import forward
from numpy import eye
import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
from layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
DTYPE = torch.float32


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
