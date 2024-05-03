# https://github.com/steveazzolin/gnn_logic_global_expl/blob/master/datasets/BAMultiShapes/generate_dataset.py

from networkx.generators import random_graphs, lattice, small, classic
import networkx as nx
import random
import numpy as np


def merge_graphs(g1, g2, nb_random_edges=1):
    mapping = dict()
    max_node = max(g1.nodes())

    i = 1
    for n in g2.nodes():
        mapping[n] = max_node + i
        i = i + 1
    g2 = nx.relabel_nodes(g2,mapping)

    g12 = nx.union(g1,g2)
    for i in range(nb_random_edges):
        e1 = list(g1.nodes())[random.randint(0,len(g1.nodes())-1)]
        e2 = list(g2.nodes())[random.randint(0,len(g2.nodes())-1)]
        g12.add_edge(e1,e2)        
    return g12

def generate_class1(nb_random_edges, nb_node_ba=40, seed=0):
    r = np.random.randint(3)
    
    if r == 0: # W + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-9, 1, seed=seed)
        g2 = classic.wheel_graph(6)
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = merge_graphs(g12,g3,nb_random_edges)
    elif r == 1: # W + H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-5, 1, seed=seed)
        g2 = classic.wheel_graph(6)
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = small.house_graph()
        g123 = merge_graphs(g12,g3,nb_random_edges)
    elif r == 2: # H + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5-9, 1, seed=seed)
        g2 = small.house_graph()
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = merge_graphs(g12,g3,nb_random_edges)
    return g123

def generate_class0(nb_random_edges, nb_node_ba=40, seed=0):
    r = np.random.randint(10)
    
    if r > 3:
        g12 = random_graphs.barabasi_albert_graph(nb_node_ba, 1, seed=seed) 
    if r == 0: # W
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6, 1, seed=seed)
        g2 = classic.wheel_graph(6)
        g12 = merge_graphs(g1,g2,nb_random_edges)      
    if r == 1: # H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5, 1, seed=seed)
        g2 = small.house_graph()
        g12 = merge_graphs(g1,g2,nb_random_edges)      
    if r == 2: # G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9, 1, seed=seed)
        g2 = lattice.grid_2d_graph(3, 3)
        g12 = merge_graphs(g1,g2,nb_random_edges)            
    if r == 3: # All
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9-5-6, 1, seed=seed)
        g2 = small.house_graph()
        g12 = merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = merge_graphs(g12,g3,nb_random_edges)
        g4 =  classic.wheel_graph(6)
        g12 = merge_graphs(g123,g4,nb_random_edges)
    return g12

def generate(num_samples, seed=0):
    assert num_samples % 2 == 0
    adjs = []
    labels = []
    feats = []
    nb_node_ba = 40

    for _ in range(int(num_samples/2)):
        g = generate_class1(nb_random_edges=1, nb_node_ba=nb_node_ba, seed=seed)
        adjs.append(nx.adjacency_matrix(g).toarray())
        labels.append(0)
        feats.append(list(np.ones((len(g.nodes()),10))/10))

    for _ in range(int(num_samples/2)):
        g = generate_class0(nb_random_edges=1, nb_node_ba=nb_node_ba, seed=seed)
        adjs.append(nx.adjacency_matrix(g).toarray())
        labels.append(1)
        feats.append(list(np.ones((len(g.nodes()), 10))/10))
    return adjs,feats,labels 
