from itertools import islice
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear, Sequential, Tanh, Sigmoid
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool, GraphNorm
from lightning import LightningModule


def gin_conv(in_channels, out_channels):
    nn = Sequential(Linear(in_channels, 2 * in_channels), GraphNorm(2 * in_channels), ReLU(), Linear(2 * in_channels, out_channels))
    return GINConv(nn)


class GNN(LightningModule):
    def __init__(self, num_features, num_classes, layers, dim, conv="GCN", activation="ReLU", pool="mean", lr=1e-4, weight=None):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.layers = layers
        self.dim = dim
        self.conv = conv
        self.pool = pool
        self.lr = lr
        self.conv_name = conv

        self.embedding = Linear(num_features, dim)
        match activation:
            case "ReLU": self.act = ReLU()
            case "Tanh": self.act = Tanh()
            case "Sigmoid": self.act = Sigmoid()
            case _: raise ValueError(f"Unknown activation {activation}")
        
        match conv:
            case "GCN": self.conv = GCNConv
            case "GIN": self.conv = gin_conv    
        
        
        self.conv = GCNConv
        self.norms = torch.nn.ModuleList(
            [GraphNorm(dim) for _ in range(layers)])
        self.conv_layers = torch.nn.ModuleList(
            [self.conv(dim, dim) for _ in range(layers)])
        self.out = Sequential(
            Linear(dim, dim),
            self.act,
            Linear(dim, num_classes)
        )
        self.loss = torch.nn.NLLLoss(weight=weight)
        
        self.save_hyperparameters('num_features', 'num_classes', 'layers', 'dim', 'activation', 'pool', 'lr', 'weight')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        for conv, norm in zip(self.conv_layers, self.norms):
            x = x + norm(conv(x, edge_index))
            x = self.act(x)
        match self.pool:
            case "mean": x = global_mean_pool(x, batch)
            case "add": x = global_add_pool(x, batch)
            case _: raise ValueError(f"Unknown aggregation {self.pool}")
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean().item()
        f1_macro = f1_score(batch.y.cpu(), out.argmax(dim=1).cpu(), average='macro')
        self.log(f'{self.conv_name}_train_loss', loss, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_train_acc', acc, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_train_f1_macro', f1_macro, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean().item()
        f1_macro = f1_score(batch.y.cpu(), out.argmax(dim=1).cpu(), average='macro')
        self.log(f'{self.conv_name}_val_loss', loss, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_val_acc', acc, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log(f'{self.conv_name}_val_f1_macro', f1_macro, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        return loss
