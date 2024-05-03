from itertools import islice
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear, Sequential, Tanh, Sigmoid
from torch_geometric.nn import GATv2Conv, GCNConv, global_add_pool, global_mean_pool, GraphNorm
from lightning import LightningModule


class GNN(LightningModule):
    def __init__(self, num_features, num_classes, layers, dim, activation="Sigmoid", lr=1e-4, dropout=0.0, weight=None):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dim = dim
        self.layers = layers
        self.type = type
        self.lr = lr

        self.embedding = Linear(num_features, dim)
        match activation:
            case "ReLU": self.act = ReLU()
            case "Tanh": self.act = Tanh()
            case "Sigmoid": self.act = Sigmoid()
            case _: raise ValueError(f"Unknown activation {activation}")
        
        self.dropout = torch.nn.Dropout(dropout)
        self.conv = GCNConv
        self.norms = torch.nn.ModuleList(
            [GraphNorm(dim) for _ in range(layers)])
        self.conv_layers = torch.nn.ModuleList(
            [self.conv(dim, dim) for _ in range(layers)])
        self.out = Linear(2 * dim, num_classes)
        self.loss = torch.nn.NLLLoss(weight=weight)
        
        self.save_hyperparameters('num_features', 'num_classes', 'layers', 'dim', 'activation', 'lr', 'dropout', 'weight')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        for conv, norm in zip(self.conv_layers, self.norms):
            x = x + self.dropout(norm(conv(x, edge_index)))
            x = self.act(x)
        x = torch.concat((global_mean_pool(x, batch), global_add_pool(x, batch)), dim=1)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean().item()
        self.log('train_loss', loss, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log('train_acc', acc, on_step=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean().item()
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        self.log('val_acc', acc, on_epoch=True, logger=True, prog_bar=False, sync_dist=True, batch_size=batch.num_graphs)
        return loss


class ConvWrapper(torch.nn.Module):
    def __init__(self, conv, nn, norm, act, dropout):
        super(ConvWrapper, self).__init__()
        self.conv = conv
        self.nn = nn
        self.act = act
        self.norm = norm
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        return self.nn(self.act(x + self.dropout(self.norm(self.conv(x, edge_index)))))