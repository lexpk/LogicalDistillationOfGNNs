from scipy.sparse import coo_matrix

import networkx as nx
from networkx.generators import random_graphs, lattice, small, classic
import numpy as np

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset


def data(name, kfold, cv_split, seed=0):
    rng = np.random.default_rng(seed)
    if name in ['EMLC0', 'EMLC1', 'EMLC2', 'BAMultiShapes']:
        if name == 'BAMultiShapes':
            datalist = [_generate_BAMultiShapes(rng) for _ in range(8000)]
        else:
            datalist = [_generate_EMLC(name, rng) for _ in range(5000)]
        
        n_test = len(datalist) // kfold
        
        train_val_data = datalist[:cv_split * n_test] + datalist[(cv_split + 1) * n_test:]
        train_data, val_data = torch.utils.data.random_split(train_val_data, [len(train_val_data) - n_test, n_test])
        test_data = datalist[cv_split * n_test : (cv_split + 1) * n_test]

        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

        train_val_batch = Batch.from_data_list(train_val_data)
        test_batch = Batch.from_data_list(test_data)

        return 1, 2, train_loader, val_loader, train_val_batch, test_batch
    else:
        dataset = TUDataset(root='data', name=name)
        n_test = len(dataset) // kfold
        permutation = torch.tensor(rng.permutation(len(dataset)))

        train_val_data = dataset[torch.concat([permutation[:cv_split * n_test], permutation[(cv_split + 1) * n_test:]])]
        train_data, val_data = torch.utils.data.random_split(train_val_data, [len(train_val_data) - n_test, n_test])
        test_data = dataset[permutation[cv_split * n_test : (cv_split + 1) * n_test]]

        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        
        train_val_batch = Batch.from_data_list(train_val_data)
        test_batch = Batch.from_data_list(test_data)
        
        return dataset.num_features, dataset.num_classes, train_loader, val_loader, train_val_batch, test_batch
    

def _generate_EMLC(name, rng):
    u0 = (rng.random((13, 1)) < 0.5).astype(np.float32)
    u1 = np.ones((13, 1), dtype=np.float32)
    graph = nx.erdos_renyi_graph(13, 0.5, seed=rng.choice(2**32))
    adj = nx.adjacency_matrix(graph).toarray()
    edge_index = _adj_to_edge_index(adj)

    match name:
        case 'EMLC0':
            has_more_than_half_u0 = (u0.sum() > 6)
            return Data(x=torch.tensor(u0), edge_index=_adj_to_edge_index(adj), y=int(has_more_than_half_u0))
        case 'EMLC1':
            has_lt_4_or_gt_9_neighbors = (edge_index[0].bincount() < 4) | (edge_index[0].bincount() > 9)
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=int(has_lt_4_or_gt_9_neighbors.max()))
        case 'EMLC2':
            degrees = adj.sum(1)
            has_gt_6_neighbors = degrees > 6
            more_than_half_neighbours_with_gt_6_neighbors = ((adj @ has_gt_6_neighbors) / degrees.clip(1)) > 0.5
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=(torch.tensor(more_than_half_neighbours_with_gt_6_neighbors).float().mean() > 0.5).long())


# The following code is a modified version of
# https://github.com/steveazzolin/gnn_logic_global_expl/blob/master/datasets/BAMultiShapes/generate_dataset.py
def _merge_graphs(g1, g2, nb_random_edges=1, rng=np.random.default_rng(0)):
    mapping = dict()
    max_node = max(g1.nodes())

    i = 1
    for n in g2.nodes():
        mapping[n] = max_node + i
        i = i + 1
    g2 = nx.relabel_nodes(g2,mapping)

    g12 = nx.union(g1,g2)
    for i in range(nb_random_edges):
        e1 = list(g1.nodes())[rng.choice(len(g1.nodes()))]
        e2 = list(g2.nodes())[rng.choice(len(g2.nodes()))]
        g12.add_edge(e1,e2)
    return g12


def _generate_class1(nb_random_edges, nb_node_ba, rng):
    r = rng.choice(3) 
    
    if r == 0: # W + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-9, 1, seed=rng.choice(2**32))
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1,g2,nb_random_edges, rng)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12,g3,nb_random_edges)
    elif r == 1: # W + H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-5, 1, seed=rng.choice(2**32))
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1,g2,nb_random_edges, rng)
        g3 = small.house_graph()
        g123 = _merge_graphs(g12,g3,nb_random_edges, rng)
    elif r == 2: # H + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5-9, 1, seed=rng.choice(2**32))
        g2 = small.house_graph()
        g12 = _merge_graphs(g1,g2,nb_random_edges, rng)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12,g3,nb_random_edges, rng)
    return g123


def _generate_class0(nb_random_edges, nb_node_ba, rng):
    r = rng.choice(4)
    
    if r > 3:
        g12 = random_graphs.barabasi_albert_graph(nb_node_ba, 1, seed=rng.choice(2**32)) 
    if r == 0: # W
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6, 1, seed=rng.choice(2**32)) 
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1,g2,nb_random_edges)      
    if r == 1: # H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5, 1, seed=rng.choice(2**32)) 
        g2 = small.house_graph()
        g12 = _merge_graphs(g1,g2,nb_random_edges)      
    if r == 2: # G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9, 1, seed=rng.choice(2**32)) 
        g2 = lattice.grid_2d_graph(3, 3)
        g12 = _merge_graphs(g1,g2,nb_random_edges)            
    if r == 3: # All
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9-5-6, 1, seed=rng.choice(2**32)) 
        g2 = small.house_graph()
        g12 = _merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12,g3,nb_random_edges)
        g4 =  classic.wheel_graph(6)
        g12 = _merge_graphs(g123,g4,nb_random_edges)
    return g12


def _generate_BAMultiShapes(rng):
    nb_node_ba = 40
    r = rng.choice(2)
    
    if r == 0:
        g = _generate_class1(nb_random_edges=1, nb_node_ba=nb_node_ba, rng=rng)
        return Data(x=torch.ones((len(g.nodes()), 1)), edge_index=_adj_to_edge_index(nx.adjacency_matrix(g).toarray()), y=0)
    else:
        g = _generate_class0(nb_random_edges=1, nb_node_ba=nb_node_ba, rng=rng)
        return Data(x=torch.ones((len(g.nodes()), 1)), edge_index=_adj_to_edge_index(nx.adjacency_matrix(g).toarray()), y=1)


def _adj_to_edge_index(adj):
    matrix = coo_matrix(adj)
    return torch.stack([torch.tensor(matrix.row, dtype=torch.long), torch.tensor(matrix.col, dtype=torch.long)])
