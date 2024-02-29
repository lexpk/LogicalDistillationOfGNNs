import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GC2GNN(nn.Module):
    def __init__(self, node_features, encoding_dim, n_classes, n_blocks, n_layers, lamb):
        super().__init__()
        self.lamb = lamb
        self.node_dim = node_features
        self.encoding_dim = encoding_dim
        self.n_classes = n_classes
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(GC2GNNBlock(encoding_dim, n_layers, self.lamb))
        self.encoder = IMLPLayer(node_features, encoding_dim, lamb=self.lamb)
        self.decoder = IMLPLayer(encoding_dim, n_classes, lamb=self.lamb)
        self.parameter_count = sum(p.numel() for p in self.parameters())

    def forward(self, x, edge_index):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return self.decoder(x)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return self.decoder(x)

    def regularizer(self):
        return torch.sum(
            torch.stack([
                block.regularizer()
                for block in self.blocks
            ])
        ) + \
        self.encoder.regularizer() + \
        self.decoder.regularizer()
        

class GC2GNNBlock(nn.Module):
    def __init__(self, encoding_dim, n_layers, lamb):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.hidden_layers.append(IMLPLayer(encoding_dim, encoding_dim, lamb))
        self.aggregation_layer = ISumAggregationLayer(encoding_dim, lamb)

    def forward(self, x, edge_index):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.aggregation_layer(x, edge_index)

    def regularizer(self):
        return torch.sum(
            torch.stack([
                layer.regularizer()
                for layer in self.hidden_layers
            ])
        ) + self.aggregation_layer.regularizer()


class ISumAggregationLayer(gnn.MessagePassing):
    def __init__(self, encoding_dim, lamb):
        super().__init__(aggr='add')
        self.lamb = lamb
        self.bias = nn.Parameter(torch.zeros(encoding_dim))
        self.linear = nn.Linear(2*encoding_dim, encoding_dim, bias=True)
        self.act = IReLU(self.lamb)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def update(self, aggr_out, x):
        return self.act(self.linear(torch.cat([aggr_out, x], dim=-1)))

    def regularizer(self):
        a = one_of(1, 0, -1, x=self.linear.weight)
        b = torch.abs(1 - torch.abs(torch.sum(self.linear.weight, dim=1))).sum()
        return a + b

class IMLPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, lamb):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.lamb = lamb
        self.act = IReLU(self.lamb)

    def forward(self, x):
        return self.act(self.linear(x))

    def regularizer(self):
        return one_of(1, 0, -1, x=self.linear.weight)


class IReLU(nn.Module):
    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb
        
    def forward(self, x):
        return torch.max(torch.zeros_like(x), torch.min((1 + self.lamb) * x, 1 + x / (1 + self.lamb)))

    def regularizer(self):
        return torch.tensor(0)

def one_of(*args, x):
    return torch.pow(
        torch.prod(
            torch.stack([
                torch.abs(x - arg * torch.ones_like(x))
                for arg in args
            ]), dim=0
        ), 1/len(args)
    ).sum()