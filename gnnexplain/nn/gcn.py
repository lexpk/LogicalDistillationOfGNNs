import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from lightning import LightningModule


epochs=200
lr=0.01
weight_decay=0.0005
hidden=16
dropout=0.5
normalize_features=True
logger=None
optimizer='Adam'
preconditioner=None
momentum=0.9
epslt=0.01
update_freq=50
gamma=None
alpha=None
hyperparam=None

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv2 = GCNConv(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class Model(LightningModule):
    def __init__(self, dataset):
        super().__init__()
        self.model = Net(dataset)
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.save_hyperparameters()

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        out = self(batch)[batch.train_mask]
        y_train = batch.y[batch.train_mask]
        loss = self.loss(out, y_train)
        acc = (out.argmax(dim=1) == y_train).sum().item() / y_train.size(0)
        self.log('train_loss', loss, on_step=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)[batch.val_mask]
        y_val = batch.y[batch.val_mask]
        loss = self.loss(out, y_val)
        acc = (out.argmax(dim=1) == y_val).sum().item() / y_val.size(0)
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, prog_bar=True, logger=True)
        return loss
