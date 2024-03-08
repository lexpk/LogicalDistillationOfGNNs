from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import wandb

from gnnexplain.nn.gcn import Model

torch.set_float32_matmul_precision('medium')

cora = datasets.Planetoid(
    root='data',
    name='Cora',
    split='random',
    num_train_per_class=200,
    num_test=200,
    num_val=200
)

loader = DataLoader(cora, batch_size=2, shuffle=False, num_workers=4)
model = Model(cora)
logger = WandbLogger(project="gnnexplain", group="cora")

trainer = Trainer(
    max_epochs=50,
    log_every_n_steps=1,
    logger=logger,
    callbacks=[ModelCheckpoint(
        dirpath="checkpoints",
        filename="cora" + "-{epoch:02d}",
    )]
)
trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
wandb.finish()
