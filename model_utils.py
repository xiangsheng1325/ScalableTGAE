import abc
from cmath import sqrt
from itertools import count
from os import stat
from platform import node
from random import sample
import time
# from tkinter import N

from pandas import array
from models import *
from data_utils import scores_matrix_from_transition_matrix, graph_from_scores, edge_overlap, link_prediction_performance
from eval_utils import compute_graph_statistics
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import csr_matrix

#from cell import utils
#from cell.graph_statistics import compute_graph_statistics

#DEVICE = "cuda:0"
DTYPE = torch.float32


def csr_mat_to_adj_mat(csr_mat):
    t_node_num = csr_mat.shape[0]
    node_num = csr_mat.shape[1]
    # t_num = (t_node_num + 1) // node_num
    tmp_csr = csr_mat.toarray()
    # 转为7k*7*7
    tmp_adj = tmp_csr.reshape(-1, node_num, node_num)
    # print("tmp_adj: ", tmp_adj)
    
    # 转为5w*5w
    adj_mat = np.zeros((t_node_num, t_node_num))
    # adj_mat = [[0 if i // node_num != j // node_num else tmp_adj[i // node_num][i % node_num][j % node_num] for j in range(t_node_num)] for i in range(t_node_num)]
    for i in range(t_node_num):
        for j in range(t_node_num):
            if i // node_num == j // node_num:
                adj_mat[i][j] = tmp_adj[i // node_num][i % node_num][j % node_num]
    # print(adj_mat[0])
    return csr_matrix(adj_mat)


def compare_graph_stats(A, B):
    stats_A = []
    stats_B = []
    graphs_A = A.toarray().reshape(-1, A.shape[1], A.shape[1])
    graphs_B = B.toarray().reshape(-1, B.shape[1], B.shape[1])
    for i in range(len(graphs_A)):
        # 把原图变成无向图，加单位阵
        # graphs_A[i] = graphs_A[i] + graphs_A[i].T + np.eye(graphs_A[i].shape[0])
        # graphs_A[i][graphs_A[i] > 1] = 1
        stats_A.append(compute_graph_statistics(csr_matrix(graphs_A[i])))
        # stats_A.append(compute_graph_statistics(csr_matrix(graphs_A[i])))
        stats_B.append(compute_graph_statistics(csr_matrix(graphs_B[i])))
    stat = dict()
    for k in stats_A[0].keys():
        v = 0
        for i in range(len(stats_A)):
            if (stats_A[i][k] != 0):
                v += np.abs(stats_A[i][k] - stats_B[i][k]) / stats_A[i][k]
        v /= len(graphs_A)
        stat[k] = v
    return stat


class Callback(abc.ABC):
    """Abstract Class to customize train loops.
    Attributes:
        invoke_every(int): The number of calls required to invoke the Callback once.
    """

    def __init__(self, invoke_every):
        self._training_stopped = False
        self.invoke_every = invoke_every

    def __call__(self, loss, model):
        if model.step % self.invoke_every == 0:
            self.invoke(loss, model)

    def stop_training(self):
        self._training_stopped = True

    @abc.abstractmethod
    def invoke(self):
        pass


class EdgeOverlapCriterion(Callback):
    """Tracks the edge overlap and stops the training if the limit is met.
    Attributes:
        invoke_every(int): The number of calls required to invoke the Callback once.
        edge_overlap_limit(float): Stops the training if the models edge overlap reaches this limit.
    """

    def __init__(self, invoke_every, edge_overlap_limit=1.0):
        super().__init__(invoke_every)
        self.edge_overlap_limit = edge_overlap_limit

    def invoke(self, loss, model):
        """Computes the edge overlap and prints the actual step, loss, edge overlap and total time.
        It also stops the training if the computed edge overlap reaches self.edge_overlap_limit.
        Args:
            loss(float): The latest loss value.
            model(Cell): The instance of the model being trained.
        """
        start = time.time()
        model.update_scores_matrix()
        sampled_graph = model.sample_graph()
        overlap = edge_overlap(model.A_sparse, sampled_graph) / model.num_edges
        overlap_time = time.time() - start
        model.total_time += overlap_time
        # print("sampled_graph: ", sampled_graph)
        # print("test: ", compute_graph_statistics(csr_mat_to_adj_mat(sampled_graph)))

        step_str = f"{model.step:{model.step_str_len}d}"
        print(
            f"Step: {step_str}/{model.steps}",
            f"Loss: {loss:.5f}",
            f"Edge-Overlap: {overlap:.3f}",
            f"Total-Time: {int(model.total_time)}",
        )
        if overlap >= self.edge_overlap_limit:
            self.stop_training()


class LinkPredictionCriterion(Callback):
    """Evaluates the link prediction performance and stops the training if there is no improvement for several steps.
    
    It ensures that the model's score_matrix is set to the score_matrix yielding the best results so far.
    
    Attributes:
        invoke_every(int): The number of calls required to invoke the Callback once.
        edge_overlap_limit(float): Stops the training if the models edge overlap reaches this limit.
        val_ones(np.ndarray): Validation ones for link prediction.
        val_zeros(np.ndarray): Validation zeros for link prediction.
        max_patience(int): Maximal number of invokes without improvement of link prediction performance
            until the training is stopped.
    """

    def __init__(self, invoke_every, val_ones, val_zeros, max_patience):
        super().__init__(invoke_every)
        self.val_ones = val_ones
        self.val_zeros = val_zeros
        self.max_patience = max_patience

        self._patience = 0
        self._best_scores_matrix = None
        self._best_link_pred_score = 0.0

    def invoke(self, loss, model):
        """Evaluates the link prediction performance and prints the actual step, loss, edge overlap and total time.
        It also stops the training if there is no improvement for self.max_patience invokes.
        Args:
            loss(float): The latest loss value.
            model(Cell): The instance of the model being trained.
        """
        start = time.time()
        model.update_scores_matrix()
        roc_auc, avg_prec = link_prediction_performance(
            model._scores_matrix, self.val_ones, self.val_zeros
        )

        link_pred_time = time.time() - start
        model.total_time += link_pred_time

        step_str = f"{model.step:{model.step_str_len}d}"
        print(
            f"Step: {step_str}/{model.steps}",
            f"Loss: {loss:.5f}",
            f"ROC-AUC Score: {roc_auc:.3f}",
            f"Average Precision: {avg_prec:.3f}",
            f"Total-Time: {int(model.total_time)}",
        )
        link_pred_score = roc_auc + avg_prec

        if link_pred_score > self._best_link_pred_score:
            self._best_link_pred_score = link_pred_score
            self._best_scores_matrix = model._scores_matrix.copy()
            self._patience = 0

        elif self._patience >= self.max_patience:
            self.stop_training()
        else:
            self._patience += 1
        model._scores_matrix = self._best_scores_matrix


class StatisticCollector(Callback):
    """Tracks graph statistics.
    Attributes:
        invoke_every(int): The number of calls required to invoke the Callback once.
        A(sp.csr_matrix): input graph.
        test_ones(np.ndarray): Test ones for link prediction.
        test_zeros(np.ndarray): Test zeros for link prediction.
        graphic_mode(str): x-axis for the graph plotting.
        n_samples(int): number of generated graphs for averaging
    """
    def __init__(self, invoke_every, A, test_ones, test_zeros, graphic_mode='overlap', n_samples=5, evolve=False, g_path=None):
        super().__init__(invoke_every)
        self.invoke_every = invoke_every
        self.test_ones = test_ones
        self.test_zeros = test_zeros
        self.n_samples = n_samples
        self.A = A
        self.num_edges = self.A.sum() / 2
        self.graphic_mode = graphic_mode
        self.training_stat = []
        self.evolve = evolve
        self.g_path = g_path

    def invoke(self, loss, model, last=False, self_loop=False):
        """Evaluates the link prediction performance and collect graph statistics.
        Args:
            loss(float): The latest loss value - not needed.
            model(Cell): The instance of the model being trained.
        """
        model.update_scores_matrix()
        roc_auc, avg_prec = link_prediction_performance(scores_matrix=model._scores_matrix, 
                                val_ones=self.test_ones, 
                                val_zeros=self.test_zeros)

        generated_graphs = [model.sample_graph() for _ in range(self.n_samples)]
        # current_overlaps = [edge_overlap(self.A, gg) / model.num_edges for gg in generated_graphs]
        current_overlaps = [edge_overlap(self.A, gg) / self.num_edges for gg in generated_graphs]
        current_overlap = np.mean(current_overlaps)
        
        # print("generated_graphs: ", generated_graphs)
        # stats = [compute_graph_statistics(csr_mat_to_adj_mat(gg)) for gg in generated_graphs]
        # comp_A = self.A.toarray().reshape(-1, self.A.shape[1], self.A.shape[1])
        # for i in range(len(comp_A)):
        #     comp_A[i] = comp_A[i] + comp_A[i].T + np.eye(comp_A[i].shape[0])
        # comp_A = sp.csr_matrix(comp_A.reshape(self.A.shape[0], self.A.shape[1]))
        # comp_A[comp_A > 1] = 1
        
        # For filtered graph
        # N = generated_graphs[0].shape[1]
        # T = len(nonzero_rows) // N
        # count_rows = np.array([np.count_nonzero(nonzero_rows[N * t : N * t + 1]) for t in range(T)])
        # assert count_rows.sum() == generated_graphs[0].shape[0]
        # for i in range(len(generated_graphs)):
        #     tmp_g = np.zeros((N * T, N))
        #     for t in range(T):
        #         generated_graphs[i][count_rows[0:t].sum() : count_rows[0:t+1].sum()] &= nonzero_rows[N * t : N * t + 1]
        #     tmp_g[np.nonzero(nonzero_rows)] = generated_graphs[i]
        #     tmp_g = tmp_g + tmp_g.T
        #     tmp_g[tmp_g > 1] = 1
        #     generated_graphs[i] = tmp_g

        generated_graphs = np.array([gg.toarray().reshape(-1, gg.shape[1], gg.shape[1]) for gg in generated_graphs])

        # For evolve edges
        if self.evolve:
            for i in range(len(generated_graphs)):
                if self_loop:
                    generated_graphs[i][0] = generated_graphs[i][0] + np.eye(generated_graphs[i][0].shape[0])
                    generated_graphs[i][0][generated_graphs[i][0] > 1] = 1
                for j in range(1, len(generated_graphs[i])):
                    generated_graphs[i][j] = generated_graphs[i][j - 1] + generated_graphs[i][j]
                    generated_graphs[i][j][generated_graphs[i][j] > 1] = 1
        
        # Save graphs
        print("g_graph:",self.g_path)
        print("last:",last)
        if last and self.g_path:
            np.save(self.g_path, np.array(generated_graphs))

        stats = [[compute_graph_statistics(csr_matrix(i)) for i in gg] for gg in generated_graphs]
        # stats = [{"Placeholder": 0.} for gg in generated_graphs]
        # for i, x in enumerate(stats):
        #     x['ROC-AUC'] = roc_auc
        #     x['AVG-PREC'] = avg_prec
        #     x['EO'] = current_overlaps[i]
        # print("stats: ", stats)
        if self.graphic_mode == 'overlap':
            self.training_stat.append({'overlap': current_overlap, 'stats': stats})
        else:
            self.training_stat.append({'iteration': model.step, 'stats': stats})


class Cell(object):
    """Implements the Cross Entropy Low-rank Logits graph generative model.
    Attributes:
        A(torch.tensor): The adjacency matrix representing the target graph.
        A_sparse(sp.csr.csr_matrix): The sparse representation of A.
        H(int): The maximum rank of W.
        loss_fn(function): The loss function minimized during the training process.
        callbacks(list): A list containing instances of classes derived from Callback.
        step(int): Keeps track of the actual training step.
        num_edges(int): The total number of edges in A.
    """

    def __init__(self, A, H, loss_fn=None, g_type='cell', callbacks=[], device='cpu', T=1, pretrained_emb=None, mask_mat=None, edge_num = None, directed=False, self_loop=False):
        self.num_edges = A.sum() / 2
        self.A_sparse = A
        self.A = torch.FloatTensor(A.toarray()).to(device)
        self.step = 1
        self.callbacks = callbacks
        self._optimizer = None
        self.device = device
        self.mask_mat = mask_mat
        self.mask_mat_t = torch.BoolTensor(mask_mat).to(device)
        self.edge_num = edge_num
        self.directed = directed
        self.self_loop = self_loop
        N_all = A.shape[0]
        N = A.shape[1]
        gamma = np.sqrt(2 / (N_all + H))

        if g_type == 'temporal':
            self.g = G_temporal(N_all, N, T, H).to(device)
        else:
            raise NameError
        #self.g = self.g.to(device)
        if loss_fn == 'local_cell':
            self.loss_fn = self.local_loss
            self.mask = self._compute_mask_for_local_loss(self.A_sparse)
        elif loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = self.built_in_loss_fn

        self.total_time = 0
        self.scores_matrix_needs_update = True

    def __call__(self):
        """Computes the learned random walk transition matrix.
        
        Returns:
            (np.array): The learned random walk transition matrix of shape(N,N)
        """
        return torch.nn.functional.softmax(self.get_W(), dim=-1).detach().cpu().numpy()

    def get_W(self):
        """Computes the logits of the learned random walk transition matrix.
        
        Returns:
            W(torch.tensor): Logits of the learned random walk transition matrix of shape(N,N)
        """
        W = self.g()
        # W = self.g(self.A)

        return W

    def built_in_loss_fn(self, W, A, num_edges):
        """Computes the weighted cross-entropy loss in logits with weight matrix.
        Args:
            W(torch.tensor): Logits of learnable (low rank) transition matrix.
            A(torch.tensor): The adjaceny matrix representing the target graph.
            num_edges(int): The total number of edges of the target graph.
            
        Returns:
            (torch.tensor): Loss at logits.
        """
        # Mask all zero rows and cols
        W[self.mask_mat_t] = 0.
        A[self.mask_mat_t] = 0.
        # W = W.masked_fill(self.mask_mat_t, value=0)
        # A = A.masked_fill(self.mask_mat_t, value=0)

        d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        # print(A.shape, d.shape, W.shape)
        loss = 0.5 * torch.sum(A * (d - W)) / num_edges
        return loss

    def _compute_mask_for_local_loss(self, A_sparse):
        """
        Computes mask (indicator) for local_loss
        Args:
            A_sparse (scipy.sparse.csr_matrix) : adjacency matrix in compressed sparse row format
        Returns:
            mask (torch.tensor): resulting mask tensor
        """
        if not isinstance(A_sparse, scipy.sparse.csr.csr_matrix):
            A_sparse = scipy.sparse.csr_matrix(A_sparse)
        dists = sp.csgraph.shortest_path(csgraph=A_sparse, directed=False)
        return torch.tensor([dists[i, :] <= i for i in range(dists.shape[0])], dtype=int)
        #return torch.tensor(dists <= 4, dtype=int)

    def local_loss(self, W, A, num_edges):
        """Computes the LOCAL weighted cross-entropy loss in logits with weight matrix.
        Args:
            W(torch.tensor): Logits of learnable (low rank) transition matrix.
            A(torch.tensor): The adjaceny matrix representing the target graph.
            num_edges(int): The total number of edges of the target graph.
        Returns:
            (torch.tensor): Loss at logits.
        """
        d = torch.log(torch.exp(W).sum(dim=-1, keepdims=True))
        return 0.5 * (torch.sum(A * (d - W)) / num_edges + torch.sum(self.mask * A * (d - W)) / num_edges)

    def _closure(self):
        W = self.get_W()
        loss = self.loss_fn(W=W, A=self.A, num_edges=self.num_edges) + self.g.add_loss()
        self._optimizer.zero_grad()
        loss.backward()
        return loss

    def _train_step(self):
        """Performs and times one optimization step."""
        time_start = time.time()
        loss = self._optimizer.step(self._closure)
        time_end = time.time()
        return loss.item(), (time_end - time_start)

    def train(self, steps, optimizer_fn, optimizer_args, EO_criterion=None):
        """Starts the train loop.
        """
        self._optimizer = optimizer_fn(self.g.parameters(), **optimizer_args)
        self.steps = steps
        self.step_str_len = len(str(steps))
        self.scores_matrix_needs_update = True
        stop = False
        for self.step in range(self.step, steps + self.step):
            loss, time = self._train_step()
            self.total_time += time
            for callback in self.callbacks:
                callback(loss=loss, model=self)
                stop = stop or callback._training_stopped
            if stop:
                break

    def update_scores_matrix(self):
        """Updates the score matrix according to W."""
        self._scores_matrix = scores_matrix_from_transition_matrix(
            # transition_matrix=self(), symmetric=False
            transition_matrix=self(), symmetric=not self.directed
        )
        self.scores_matrix_needs_update = False

    def sample_graph(self):
        """Samples a graph from the learned parameters W.
        
        Edges are sampled independently from the score maxtrix.
        
        Returns:
            sampled_graph(sp.csr.csr_matrix): A synthetic graph generated by the model.
        """
        if self.scores_matrix_needs_update:
            self.update_scores_matrix()

        N = self.A_sparse.shape[1]
        T = self.A_sparse.shape[0] // N
        sampled_graph = np.zeros(self.A_sparse.shape)
        for t in range(T):
            size = int(sqrt(N * N - np.count_nonzero(self.mask_mat[t * N : (t + 1) * N])).real)
            tmp_scores_matrix = self._scores_matrix[t * N : (t + 1) * N][~self.mask_mat[t * N : (t + 1) * N]].reshape((size, size))
            tmp_graph_part = graph_from_scores(tmp_scores_matrix, self.edge_num[t], self_loop=self.self_loop)
            tmp_graph = np.zeros((N, N))
            tmp_graph[~self.mask_mat[t * N : (t + 1) * N]] = tmp_graph_part.toarray().reshape(-1)
            # if self.self_loop:
            #     tmp_graph = tmp_graph + np.eye(tmp_graph.shape[0])
            #     tmp_graph[tmp_graph > 1] = 1
            sampled_graph[t * N : (t + 1) * N] = tmp_graph
            # rows, cols = (~self.mask_mat[t * N : (t + 1) * N]).nonzero()
            # for row in range(size):
            #     for col in range(size):
            #         if (tmp_graph_part[row, col] > 0):
            #             sampled_graph[t * N + rows[row * size + col], cols[row * size + col]] = 1

        return csr_matrix(sampled_graph)