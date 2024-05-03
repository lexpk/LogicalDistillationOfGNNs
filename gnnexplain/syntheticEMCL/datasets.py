import torch
from torch_geometric.data import Data
from torch_geometric.utils.random import erdos_renyi_graph


def generate(index):
    match index:
        case 1:
            edge_index = erdos_renyi_graph(100, 0.1)
            has_lt_9_or_gt_11_neighbors = (edge_index[0].bincount() < 8) | (edge_index[0].bincount() > 12)
            has_gt_40p_U1 = has_lt_9_or_gt_11_neighbors.sum() > 40
            return Data(x=torch.ones((100, 1)), edge_index=edge_index, y=int(has_gt_40p_U1.item()))
