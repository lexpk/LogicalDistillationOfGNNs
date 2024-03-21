import optuna
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.data import Batch
from lightning import Trainer
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import wandb

from optuna.integration.wandb import WeightsAndBiasesCallback
from gnnexplain.model.gtree import Explainer, OptimizingExplainer
from gnnexplain.nn.gcn import GraphGCN

# Running on different hardware or with different versions of libraries may result in different results
torch_seed, lightning_seed, numpy_seed, optuna_seed = 0, 0, 0, 0

torch.manual_seed(torch_seed)
seed_everything(lightning_seed)


torch.set_float32_matmul_precision('medium')

dataset_names = ['MUTAG', 'AIDS']

for name in dataset_names:
    dataset = datasets.TUDataset(root='data', name=name).shuffle()
    n_train = len(dataset) * 4 // 5
    train_data, val_data = dataset[:n_train], dataset[n_train:]

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=15)
    val_loader = DataLoader(val_data, batch_size=100, shuffle=False, num_workers=15)

    model = GraphGCN(dataset)
    logger = WandbLogger(project="gnnexplain", group=name)

    trainer = Trainer(
        max_epochs=100,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[ModelCheckpoint(
            dirpath="checkpoints",
            filename=name+"-{epoch:02d}",
        )],
        devices=1,
        deterministic=True
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    wandb_kwargs = {
        'project': 'gnnexplain',
        'group': name
    }
    wandbc = WeightsAndBiasesCallback(metric_name="gtree_val_acc", wandb_kwargs=wandb_kwargs, as_multirun=True)
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    train_batch = Batch.from_data_list(train_data)
    expl = OptimizingExplainer().fit(train_batch, model)
    val_batch = Batch.from_data_list(val_data)
    acc = expl.accuracy(val_batch)

    wandb.log({'gtree_val_acc': acc})
    
    wandb.finish()
