import copy

import numpy as np
import scipy.sparse as sp
import warnings
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import pandas as pd
from scipy.sparse.linalg import eigs
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
from libwon.utils import setup_seed
from igraph import Graph
def get_sbm_graph(N, pin, pout, C, directed):
    """使用随机块模型 (SBM) 生成图。

    Args:
        N (_type_): 每个社区的节点数量
        pin (_type_): 社区内部边的生成概率
        pout (_type_): 社区之间边的生成概率
        C (_type_): 社区数量
        directed (_type_): 图是否为有向图

    Returns:
        _type_: 图中的边数组，形状为 [E, 2]。
    """
    sizes = [N] * C
    in_prob = pin
    out_prob = pout
    probs = np.zeros((len(sizes), len(sizes)))
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            probs[i, j] = in_prob if i == j else out_prob
    G = nx.stochastic_block_model(sizes, probs, directed = directed)
    edges = [e for e in G.edges()]
    return np.array(edges) # [E, 2]

def get_er_graph(N, p, directed = False):
    """生成一个 E-R随机图

    Args:
        N (_type_): 节点的数量
        p (_type_): 每对节点之间生成边的概率
        directed (bool, optional): 图是否为有向图. Defaults to False.

    Returns:
        _type_: 图中的边数组
    """
    G = nx.erdos_renyi_graph(N, p, directed = directed)
    edges = [e for e in G.edges()]
    return np.array(edges) # [E, 2]

def get_er_graphs(T = 100, N = 1000, p = 0.001, directed = False, seed = 0):
    """生成多个时间步的 ER 随机图

    Args:
        T (int, optional): 时间步数量。 Defaults to 3.
        N (int, optional): 每个图中的节点数量. Defaults to 2.
        p (float, optional): 每对节点之间生成边的概率. Defaults to 0.5.
        directed (bool, optional): 图是否为有向图. Defaults to False.
        seed (int, optional): 随机种子，用于结果的可重复性. Defaults to 0.

    Returns:
        _type_: 包含所有时间步的边数组，形状为 [E, 3]，其中每条边由两个节点和一个时间步组成
    """
    setup_seed(seed)
    es = []
    for t in range(T):
        e = get_er_graph(N, p, directed)
        if len(e):
            time = np.array([t] * e.shape[0]).reshape(e.shape[0],1)
            es.append(np.concatenate([e, time], axis = -1))
    es = np.concatenate(es, axis = 0)
    return es

def get_sbm_graphs(T = 100, N = 1000, p = 0.001, C = 3, directed = False, seed = 0):
    """生成多个时间步的随机块模型 (SBM) 图

    Args:
        T (int, optional): 时间步数量. Defaults to 3.
        N (int, optional): 每个社区的节点数量. Defaults to 2.
        p (float, optional): 社区内的边生成概率. Defaults to 0.5.
        C (int, optional): 社区数量. Defaults to 3.
        directed (bool, optional): 图是否为有向图. Defaults to False.
        seed (int, optional): 随机种子，用于结果的可重复性. Defaults to 0.

    Returns:
        _type_: 包含所有时间步的边数组，形状为 [E, 3]，其中每条边由两个节点和一个时间步组成
    """
    setup_seed(seed)
    es = []
    for t in range(T):
        e = get_sbm_graph(N, p, C, directed)
        if len(e):
            time = np.array([t] * e.shape[0]).reshape(e.shape[0],1)
            es.append(np.concatenate([e, time], axis = -1))
    es = np.concatenate(es, axis = 0)
    return es

class DyGraphGenERCon:
    def sample_dynamic_graph(self, T = 100, N = 1000 , p = 0.001, directed = False, seed = 0):
        """生成一个动态ER图的样本

        Args:
            T (int, optional): 时间步数量.
            N (int, optional): 节点数量. 
            p (float, optional): 边生成概率.
            directed (bool, optional): 图是否为有向图. Defaults to False.
            seed (int, optional): 随机种子，用于结果的可重复性. Defaults to 0.

        Returns:
            _type_: 图的详细信息，包括边、节点数量、边数量、时间步等。
        """
        es = get_er_graphs(1, N, p, directed, seed)
        es[:, 2] = np.random.randint(0, T, es.shape[0])
        es = list(es)
        # 按时间步和节点编号对边进行排序
        es = sorted(es, key = lambda x: (x[2], x[0]))
        es = np.array(es)
        # 统计节点和边的信息
        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))
        # 提取 src 和 dst 中的节点编号
        src_nodes = es[:, 0]
        dst_nodes = es[:, 1]
        all_nodes = set(src_nodes).union(set(dst_nodes))

        # 找到超出范围的节点编号
        out_of_range_nodes = {node for node in all_nodes if node >= num_nodes}
        print("out_of_range:",out_of_range_nodes)
        # 找到未使用的节点编号
        used_nodes = {node for node in all_nodes if node < num_nodes}
        unused_nodes = [node for node in range(num_nodes) if node not in used_nodes]
        # 随机打乱未使用的节点编号
        #rng = np.random.default_rng(seed)
        #rng.shuffle(unused_nodes)
        # 构建映射字典
        mapping = {}
        unused_idx = 0
        for node in out_of_range_nodes:
            if unused_idx < len(unused_nodes):
                mapping[node] = unused_nodes[unused_idx]
                unused_idx += 1
            else:
                raise ValueError("未使用的节点编号不足以替换超出范围的节点编号")

        # 使用映射字典替换 src 和 dst 中的节点编号
        for i in range(len(es)):
            if es[i, 0] in mapping:
                es[i, 0] = mapping[es[i, 0]]
            if es[i, 1] in mapping:
                es[i, 1] = mapping[es[i, 1]]

        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "T": T,
                "N":N,
                "p":p,
                "directed":directed,
                "seed":seed
                }
        print("info_ercon:",es.tolist())
        print("node_num:",num_nodes)
        # 将边列表保存为txt文件
        np.savetxt('./data/SyntheticDataSets/edges_ercon.txt', es, fmt='%d', delimiter=' ')
        return info
    
class DyGraphGenSBMCon:
    def sample_dynamic_graph(self, T = 3, N = 2 , p = 0.5, C = 2, directed = False, seed = 0):
        """使用随机块模型（SBM）生成一个动态图样本

        Args:
            T (int, optional): 时间步数. 
            N (int, optional): 总节点数. 
            p (float, optional): 创建边的概率. 
            C (int, optional): 社区数量.
            directed (bool, optional): 是否为有向图. Defaults to False.
            seed (int, optional): 随机数生成的种子. Defaults to 0.

        Returns:
            _type_: _description_
        """
        setup_seed(seed)
        es = get_sbm_graph(N//C, p, p/2, C, directed)
        es = np.concatenate([es, np.zeros((es.shape[0], 1))], axis = -1).astype(int)
        # import pdb;pdb.set_trace()
        es[:, 2] = np.random.randint(0, T, es.shape[0])
        es = list(es)
        es = sorted(es, key = lambda x: (x[2], x[0]))
        es = np.array(es)

        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))

        # 提取 src 和 dst 中的节点编号
        src_nodes = es[:, 0]
        dst_nodes = es[:, 1]
        all_nodes = set(src_nodes).union(set(dst_nodes))
        # 找到超出范围的节点编号
        out_of_range_nodes = {node for node in all_nodes if node >= num_nodes}

        # 找到未使用的节点编号
        used_nodes = {node for node in all_nodes if node < num_nodes}
        unused_nodes = [node for node in range(num_nodes) if node not in used_nodes]
        # 随机打乱未使用的节点编号
        #rng = np.random.default_rng(seed)
        #rng.shuffle(unused_nodes)
        # 构建映射字典
        mapping = {}
        unused_idx = 0
        for node in out_of_range_nodes:
            if unused_idx < len(unused_nodes):
                mapping[node] = unused_nodes[unused_idx]
                unused_idx += 1
            else:
                raise ValueError("未使用的节点编号不足以替换超出范围的节点编号")

        # 使用映射字典替换 src 和 dst 中的节点编号
        for i in range(len(es)):
            if es[i, 0] in mapping:
                es[i, 0] = mapping[es[i, 0]]
            if es[i, 1] in mapping:
                es[i, 1] = mapping[es[i, 1]]

        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "T": T,
                "N":N,
                "p":p,
                "directed":directed,
                "seed":seed
                }
        #print("info_ffcon:",info)
        # 将边列表保存为txt文件
        np.savetxt('./data/SyntheticDataSets/edges_sbmcon.txt', es, fmt='%d', delimiter=' ')
        return info
    
class DyGraphGenFFCon:
    def sample_dynamic_graph(self, T, N , p, seed, directed = False):
        """生成一个动态图的样本，使用 Forest Fire 模型

        Args:
            T (int, optional): 时间步数量. 
            N (int, optional): 节点数量. 
            p (float, optional): Forest Fire 模型的前进概率. 
            directed (bool, optional): 图是否为有向图. Defaults to False.
            seed (int, optional): 随机种子，确保可重复性. Defaults to 0.

        Returns:
            _type_: 包含图信息的字典，包括边、节点和时间步数
        """
        setup_seed(seed)
        # es = get_sbm_graph(N//C, p, p/2, C, directed)
        es = Graph.Forest_Fire(N, fw_prob = p).get_edgelist()
        es = np.array(es)
        es = np.concatenate([es, np.zeros((es.shape[0], 1))], axis = -1).astype(int)
        # import pdb;pdb.set_trace()
        es[:, 2] = np.random.randint(0, T, es.shape[0])
        es = list(es)
        es = sorted(es, key = lambda x: (x[2], x[0]))
        es = np.array(es)

        num_nodes = len(set(es[:,:2].flatten()))
        num_edges = es.shape[0]
        num_time =  len(set(es[:, 2].flatten()))

        # 提取 src 和 dst 中的节点编号
        src_nodes = es[:, 0]
        dst_nodes = es[:, 1]
        all_nodes = set(src_nodes).union(set(dst_nodes))

        # 找到超出范围的节点编号
        out_of_range_nodes = {node for node in all_nodes if node >= num_nodes}

        # 找到未使用的节点编号
        used_nodes = {node for node in all_nodes if node < num_nodes}
        unused_nodes = [node for node in range(num_nodes) if node not in used_nodes]
        # 随机打乱未使用的节点编号
        #rng = np.random.default_rng(seed)
        #rng.shuffle(unused_nodes)
        # 构建映射字典
        mapping = {}
        unused_idx = 0
        for node in out_of_range_nodes:
            if unused_idx < len(unused_nodes):
                mapping[node] = unused_nodes[unused_idx]
                unused_idx += 1
            else:
                raise ValueError("未使用的节点编号不足以替换超出范围的节点编号")

        # 使用映射字典替换 src 和 dst 中的节点编号
        for i in range(len(es)):
            if es[i, 0] in mapping:
                es[i, 0] = mapping[es[i, 0]]
            if es[i, 1] in mapping:
                es[i, 1] = mapping[es[i, 1]]
       
        info = {"edge_index": es.tolist(), 
                "num_nodes":num_nodes, 
                "num_edges":num_edges, 
                "num_time": num_time, 
                "T": T,
                "N":N,
                "p":p,
                "directed":directed,
                "seed":seed
                }
        #print("info_ffcon:",info)
        # 将边列表保存为txt文件
        np.savetxt('./data/SyntheticDataSets/edges_ffcon.txt', es, fmt='%d', delimiter=' ')
        return info
    
def load_npy(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph
    """
    #if not file_name.endswith('.npz'):
    #    file_name += '.npz'
    print('file_name: ', file_name)
    loader = np.load(file_name, allow_pickle=True)
    if True:
        loader = loader.tolist()
        # 确保数据是numpy数组
        adj_data = np.array(loader['adj_data'])
        adj_indices = np.array(loader['adj_indices'])
        adj_indptr = np.array(loader['adj_indptr'])
        # 创建稀疏矩阵
        adj_matrix = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=loader['adj_shape'])
         # 调整为方阵
        #num_rows, num_cols = adj_matrix.shape
        #if num_rows != num_cols:
            #max_dim = max(num_rows, num_cols)
            #square_matrix = sp.csr_matrix((max_dim, max_dim))
            #square_matrix[:num_rows, :num_cols] = adj_matrix
            #adj_matrix = square_matrix

        if 'attr_data' in loader:
            attr_data = np.array(loader['attr_data'])
            attr_indices = np.array(loader['attr_indices'])
            attr_indptr = np.array(loader['attr_indptr'])
        
            attr_matrix = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=loader['attr_shape'])
        labels = loader.get('labels')
        return adj_matrix, attr_matrix if 'attr_data' in loader else None,labels
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph
    """
    #if not file_name.endswith('.npz'):
    #    file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def edges_to_sparse(edges, N, values=None):
    """
    Create a sparse adjacency matrix from an array of edge indices and (optionally) values.
    Parameters
    ----------
    edges : array-like, shape [n_edges, 2]
        Edge indices
    N : int
        Number of nodes
    values : array_like, shape [n_edges]
        The values to put at the specified edge indices. Optional, default: np.ones(.)
    Returns
    -------
    A : scipy.sparse.csr.csr_matrix
        Sparse adjacency matrix
    """
    if values is None:
        values = np.ones(edges.shape[0])

    return sp.coo_matrix((values, (edges[:, 0], edges[:, 1])), shape=(N, N)).tocsr()


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """
    Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    seed : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False
    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges
    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    #assert A.diagonal().sum() == 0  # no self-loops
    #assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    #is_undirected = (A != A.T).nnz == 0
    is_undirected = False
    undirected = False
    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    E = A.nnz
    N = A.shape[0]
    OUT = A.shape[1]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            #assert connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = np.array(list(nx.maximal_matching(nx.DiGraph(A))))
                not_in_cover = np.array(list(set(range(N)).difference(hold_edges.flatten())))

                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                min_size = hold_edges.shape[0] + len(not_in_cover)
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / E))

                d_nic = d[not_in_cover]

                hold_edges_d1 = np.column_stack((not_in_cover[d_nic > 0],
                                                 np.row_stack(map(np.random.choice,
                                                                  A[not_in_cover[d_nic > 0]].tolil().rows))))

                if np.any(d_nic == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, not_in_cover[d_nic == 0]].T.tolil().rows)),
                                                     not_in_cover[d_nic == 0]))
                    hold_edges = np.row_stack((hold_edges, hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = np.row_stack((hold_edges, hold_edges_d1))

            else:
                # makes sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample_0 = np.random.randint(0, N, [int(1.3 * n_test), 1])
            random_sample_1 = np.random.randint(0, OUT, [int(1.3 * n_test), 1])
            random_sample = np.concatenate([random_sample_0, random_sample_1], axis=1)

        test_zeros = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        test_zeros = np.row_stack(test_zeros)[:n_test]
        #assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def edge_overlap(A, B):
    """
    Compute edge overlap between two graphs (amount of shared edges).
    Args:
        A (sp.csr.csr_matrix): First input adjacency matrix.
        B (sp.csr.csr_matrix): Second input adjacency matrix.
    Returns:
        Edge overlap.
    """

    return A.multiply(B).sum() / 2


def link_prediction_performance(scores_matrix, val_ones, val_zeros):
    """
    Compute the link prediction performance of a score matrix on a set of validation edges and non-edges.
    Args:
        scores_matrix (np.array): Symmetric scores matrix of the graph generative model.
        val_ones (np.array): Validation edges. Rows represent indices of the input adjacency matrix with value 1. 
        val_zeros (np.array): Validation non-edges. Rows represent indices of the input adjacency matrix with value 0.
        
    Returns:
       2-Tuple containg ROC-AUC score and Average precision.
    """

    actual_labels_val = np.append(np.ones(len(val_ones)), np.zeros(len(val_zeros)))
    edge_scores = np.append(
        scores_matrix[val_ones[:, 0], val_ones[:, 1]],
        scores_matrix[val_zeros[:, 0], val_zeros[:, 1]],
    )
    return (
        roc_auc_score(actual_labels_val, edge_scores),
        average_precision_score(actual_labels_val, edge_scores),
    )


def scores_matrix_from_transition_matrix(transition_matrix, symmetric=True):
    """
    Compute the scores matrix from the transition matrix.
    Args:
        transition_matrix (np.array, shape=(N,N)).
        symmetric (bool, default:True): If True, symmetrize the resulting scores matrix.
    Returns:
        scores_matrix(sp.csr.csr_matrix, shape=(N, N)).
    """
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        return np.concatenate([i for i in transition_matrix.reshape([-1, transition_matrix.shape[1], transition_matrix.shape[1]])])
    N = transition_matrix.shape[0]
    p_stationary = np.real(eigs(transition_matrix.T, k=1, sigma=0.99999)[1])
    p_stationary /= p_stationary.sum()
    scores_matrix = np.maximum(p_stationary * transition_matrix, 0)

    if symmetric:
        scores_matrix += scores_matrix.T

    return scores_matrix


def graph_from_scores(scores_matrix, n_edges, self_loop=False):
    """
    Assemble a symmetric binary graph from the input score matrix. Ensures that there will be no singleton nodes.
    See the paper for details.
    Args:
        scores_matrix (sp.csr.csr_matrix, shape=(N, N))
        n_edges (int): The desired number of edges in the generated graph.
    Returns
    -------
    target_g (sp.csr.csr_matrix, shape=(N, N)): Adjacency matrix of the generated graph.
    """
    if scores_matrix.shape[0] != scores_matrix.shape[1]:
        per_edge = int(n_edges / scores_matrix.shape[0] * scores_matrix.shape[1])
        return sp.csr_matrix(np.concatenate([np.array(graph_from_scores(i, per_edge).todense()) for i in scores_matrix.reshape([-1, scores_matrix.shape[1], scores_matrix.shape[1]])]))
    target_g = sp.csr_matrix(scores_matrix.shape)

    if self_loop:
        np.fill_diagonal(scores_matrix, 0)

    degrees = scores_matrix.sum(1)  # The row sum over the scores_matrix.

    N = scores_matrix.shape[0]

    for n in range(N):  # Iterate over the nodes
        target = np.random.choice(N, p=scores_matrix[n] / degrees[n])
        target_g[n, target] = 1
        target_g[target, n] = 1

    diff = np.round((2 * n_edges - target_g.sum()) / 2)
    if diff > 0:
        triu = np.triu(scores_matrix)
        triu[target_g.nonzero()] = 0
        triu = triu / triu.sum()

        triu_ixs = np.triu_indices_from(scores_matrix)
        extra_edges = np.random.choice(
            triu_ixs[0].shape[0], replace=False, p=triu[triu_ixs], size=int(diff)
        )

        target_g[(triu_ixs[0][extra_edges], triu_ixs[1][extra_edges])] = 1
        target_g[(triu_ixs[1][extra_edges], triu_ixs[0][extra_edges])] = 1

    return target_g


def edge_from_scores(scores_matrix, n_edges):
    degrees = scores_matrix.sum(1)  # The row sum over the scores_matrix.
    B = scores_matrix.shape[0]
    N = scores_matrix.shape[1]
    target_g = sp.csr_matrix(scores_matrix.shape)
    probs = copy.deepcopy(scores_matrix)
    for n in range(B):  # Iterate over the nodes
        target = np.random.choice(N, p=scores_matrix[n] / degrees[n], size=1)
        target_g[n, target] = 1
        probs[n, target] = 0
    diff = np.round(n_edges - target_g.sum())
    if diff > 0:
        probs = probs.reshape(-1)
        extra_edges = np.random.choice(probs.shape[0], replace=False, p=probs/probs.sum(), size=int(diff))
        target_g[extra_edges//N, extra_edges%N] = 1
    return target_g
def main():
    FF_generator = DyGraphGenFFCon()
    FF_generator.sample_dynamic_graph(T=100, N=1000, p=0.7, seed=2025,directed=False)
    ER_generator = DyGraphGenERCon()
    ER_generator.sample_dynamic_graph(T=100, N=1000, p=0.7, seed=2025,directed=False)
    SBM_generator = DyGraphGenSBMCon()
    SBM_generator.sample_dynamic_graph(T=100, N=1000, p=0.9, seed=2025,directed=False)


if __name__ == "__main__":
    main()
    print("Success!")