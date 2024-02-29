import torch_geometric.datasets as datasets
from torch_geometric.loader import DataLoader
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

from gnnexplain.nn.model import GC2GNNModel

cora = datasets.Planetoid(
    root='data',
    name='Cora',
    split='random',
    num_train_per_class=200,
    num_test=200,
    num_val=200
)

loader = DataLoader(cora, batch_size=2, shuffle=True, num_workers=4)

torch.set_float32_matmul_precision('medium')

model = GC2GNNModel(cora.num_features, 32, cora.num_classes, 4, 4)
logger = WandbLogger(project="gnnexplain", group="cora")

trainer = Trainer(
    max_epochs=100,
    log_every_n_steps=1,
    logger=logger,
    strategy='ddp_find_unused_parameters_true'
)
trainer.fit(model, train_dataloaders=loader)
wandb.finish()