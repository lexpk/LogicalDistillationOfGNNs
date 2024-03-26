import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm
from lightning import LightningModule


class GraphGCN(LightningModule):
    def __init__(self, num_features, num_classes, hidden=256, layers=4):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden = hidden
        self.layers = layers
        
        self.act = ReLU()
        self.conv1 = GCNConv(num_features, hidden)
        self.norms = torch.nn.ModuleList(
            [GraphNorm(hidden) for _ in range(layers)])
        self.conv_layers = torch.nn.ModuleList(
            [GCNConv(hidden, hidden) for _ in range(layers)])
        self.lin = Linear(hidden, num_classes)
        self.loss = torch.nn.NLLLoss()
        
        self.save_hyperparameters('hidden', 'layers', 'num_features', 'num_classes')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.act(self.conv1(x, edge_index))
        for conv, norm in zip(self.conv_layers, self.norms):
            x = self.act(conv(norm(x), edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

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

    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean().item()
        self.log('test_loss', loss, sync_dist=True, batch_size=batch.num_graphs)
        self.log('test_acc', acc, sync_dist=True, batch_size=batch.num_graphs)
        return loss
