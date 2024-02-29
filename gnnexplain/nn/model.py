from gnnexplain.nn.acgnn import GC2GNN
from lightning import LightningModule

import torch
from torch.nn import functional as F


class GC2GNNModel(LightningModule):
    def __init__(self, node_features, encoding_dim, n_classes, n_blocks, n_layers, lamb_step=1e-1):
        super().__init__()
        self.lamb = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.lamb_step = torch.nn.Parameter(torch.tensor(lamb_step), requires_grad=False)
        self.model = GC2GNN(node_features, encoding_dim, n_classes, n_blocks, n_layers, self.lamb)
        
        self.save_hyperparameters()

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch.x, batch.edge_index)
        loss = F.cross_entropy(y_hat[batch.train_mask], batch.y[batch.train_mask])
        
        self.log("train_loss", loss,
            on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        regularizer = self.model.regularizer() / self.model.parameter_count
        self.log("regularizer", regularizer,
            on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("lamb * regularizer", self.lamb * regularizer,
            on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        return loss + self.lamb * regularizer

    def validation_step(self, batch, batch_idx):
        x, edge_index, y = batch
        y_hat = self.model(x, edge_index)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, edge_index, y = batch
        y_hat = self.model(x, edge_index)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def on_train_epoch_start(self):
        self.lamb += self.lamb_step
        return self.lamb
