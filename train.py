import optuna
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.data import Batch
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import wandb

from gnnexplain.model.gtree import Optimizer
from gnnexplain.nn.gcn import GraphGCN


torch.set_float32_matmul_precision('medium')
dataset_names = [
    'MUTAG', 'Mutagenicity', 'NCI1', 'NCI109',
    'AIDS', 'BZR', 'BZR_MD',
    'ENZYMES', 'PROTEINS',
]

k_fold = 5
for name in dataset_names:
    for iteration in range(k_fold):
        if name == 'BZR' and iteration != 4:
            continue
        dataset = datasets.TUDataset(root='data', name=name)
        n_val = len(dataset) // k_fold

        val_data = dataset[iteration * n_val : (iteration + 1) * n_val]
        train_data = dataset[:iteration * n_val] + dataset[(iteration + 1) * n_val:]

        val_loader = DataLoader(val_data, batch_size=min(len(val_data), 1000), shuffle=False, num_workers=15)
        train_loader = DataLoader(train_data, batch_size=min(len(train_data), 1000), shuffle=True, num_workers=15)
        
        if iteration == 0:
            num_classes = train_data.datasets[1].y.max() + 1
            weight = len(train_data.datasets[1]) / (num_classes * torch.bincount(train_data.datasets[1].y, minlength=num_classes).float())
        elif iteration == k_fold - 1:
            num_classes = train_data.datasets[0].y.max() + 1
            weight = len(train_data.datasets[0]) / (num_classes * torch.bincount(train_data.datasets[0].y, minlength=num_classes).float())
        else:
            num_classes = max(train_data.datasets[0].y.max(), train_data.datasets[1].y.max()) + 1
            bincount = torch.bincount(train_data.datasets[0].y, minlength=num_classes) + torch.bincount(train_data.datasets[1].y, minlength=num_classes)
            weight = (len(train_data.datasets[0])  + len(train_data.datasets[1])) / (num_classes * bincount.float())
        model = GraphGCN(dataset.num_features, dataset.num_classes, hidden=64, layers=3, weight=weight)
        logger = WandbLogger(project="gnnexplain", group=name+f'_{k_fold}fold_1')

        trainer = Trainer(
            max_steps=5000,
            log_every_n_steps=10,
            logger=logger,
            callbacks=[ModelCheckpoint(
                dirpath="checkpoints",
                filename=name+f"split={iteration}"+"-{step:02d}-{val_acc:.2%}",
                every_n_train_steps=500,
            )],
            devices=1
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        train_batch = Batch.from_data_list(train_data)
        expl = Optimizer(n_trials=100).optimize(train_batch, model, logger=logger)
        val_batch = Batch.from_data_list(val_data)
        val_acc = expl.accuracy(val_batch)

        expl.save_image('./figures/' + name + f"-{val_acc:.0%}")
        logger.log_image('explanation', images=['./figures/' + name + f"-{val_acc:.0%}.png"])
        
        logger.experiment.finish()
        
        api = wandb.Api()
        run = api.run(f"lexpk/gnnexplain/{logger.version}")
        with torch.no_grad():
            run.summary['gcn_val_acc'] = model(val_batch).argmax(-1).eq(val_batch.y).float().mean().item()
        run.summary['gtree_val_acc'] = val_acc
        run.summary.update()
