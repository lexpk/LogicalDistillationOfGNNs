import torch
from torch_geometric.data import Data
from torch_geometric.utils.random import erdos_renyi_graph


def generate(index):
    match index:
        case 0:
            edge_index = erdos_renyi_graph(13, 0.5)
            u0 = (torch.randn((13, 1)) < 0.5).float()
            u1 = torch.ones((13, 1), dtype=torch.float)
            u = torch.cat([u0, u1], dim=1)
            has_more_than_half_u0 = (u0.sum() > 6)
            return Data(x=u, edge_index=edge_index, y=int(has_more_than_half_u0))
        case 1:
            edge_index = erdos_renyi_graph(13, 0.5)
            u0 = (torch.randn((13, 1)) == 0).float()
            u1 = torch.ones((13, 1), dtype=torch.float)
            u = torch.cat([u0, u1], dim=1)
            has_lt_3_or_gt_10_neighbors = (edge_index[0].bincount() < 4) | (edge_index[0].bincount() > 9)
            return Data(x=u, edge_index=edge_index, y=int(has_lt_3_or_gt_10_neighbors.max()))
