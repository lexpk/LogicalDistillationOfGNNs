import os
import time
import optuna
import signal
import sys
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.data import Batch
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
from joblib import Parallel, delayed

from gnnexplain.model.gtree import Optimizer
from gnnexplain.nn.gnn import GNN

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AIDS', help='Name of the dataset')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation Function', choices=['ReLU', 'Tanh', 'Sigmoid'])
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--kfold', type=int, default=8, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of node embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--trials', type=int, default=1000, help='Number of Optuna trials')
    parser.add_argument('--max_depth', type=int, default=4, help='Maximum depth of the explanation tree')
    parser.add_argument('--max_ccp_alpha', type=float, default=0.01, help='Maximum ccp_alpha of the explanation tree')
    parser.add_argument('--lmb', type=float, default=0.1, help='Lambda for the regularization term')
    

    args = parser.parse_args()

    start_time = time.time()

    seed = 42

    def run(iteration, start_time, device=0):
        torch.set_float32_matmul_precision('medium')
        torch.manual_seed(seed)
        
        dataset = datasets.TUDataset(root='data', name=args.dataset)
        n_val = len(dataset) // args.kfold

        val_data = dataset[iteration * n_val : (iteration + 1) * n_val]
        train_data = dataset[:iteration * n_val] + dataset[(iteration + 1) * n_val:]

        val_loader = DataLoader(val_data, batch_size=min(len(val_data), args.batch_size), shuffle=False, num_workers=64//args.kfold, multiprocessing_context='fork')
        train_loader = DataLoader(train_data, batch_size=min(len(train_data), args.batch_size), shuffle=True, num_workers=64//args.kfold, multiprocessing_context='fork')
        
        if iteration == 0:
            num_classes = train_data.datasets[1].y.max() + 1
            weight = len(train_data.datasets[1]) / (num_classes * torch.bincount(train_data.datasets[1].y, minlength=num_classes).float())
        elif iteration == args.kfold - 1:
            num_classes = train_data.datasets[0].y.max() + 1
            weight = len(train_data.datasets[0]) / (num_classes * torch.bincount(train_data.datasets[0].y, minlength=num_classes).float())
        else:
            num_classes = max(train_data.datasets[0].y.max(), train_data.datasets[1].y.max()) + 1
            bincount = torch.bincount(train_data.datasets[0].y, minlength=num_classes) + torch.bincount(train_data.datasets[1].y, minlength=num_classes)
            weight = (len(train_data.datasets[0])  + len(train_data.datasets[1])) / (num_classes * bincount.float())
        
        model = GNN(dataset.num_features, dataset.num_classes, layers=args.layers, dim=args.dim, activation=args.activation, lr=args.lr, dropout=args.dropout, weight=weight)
        os.environ["WANDB_SILENT"] = "true"
        logger = WandbLogger(project="gnnexplain", group=f'{args.dataset}_{args.activation}_{args.kfold}fold_{args.layers}layers_{args.dim}dim_{start_time}')

        trainer = Trainer(
            max_steps=args.steps,
            logger=logger,
            devices=[device],
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint(f'./checkpoints/{args.dataset}_{args.activation}_{logger.version}.ckpt')

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        train_batch = Batch.from_data_list(train_data)
        expl = Optimizer(n_trials=args.trials, max_depth=args.max_depth, max_ccp_alpha=args.max_ccp_alpha, lmb=args.lmb).optimize(train_batch, model, logger=logger, progress_bar=False)
        val_batch = Batch.from_data_list(val_data)
        val_acc = expl.accuracy(val_batch)

        expl.save_image('./figures/' + args.dataset + f"-{val_acc:.0%}")
        logger.log_image('explanation', images=['./figures/' + args.dataset + f"-{val_acc:.0%}.png"])
        
        train_fidelity = expl.fidelity(train_batch, model)
        val_fidelity = expl.fidelity(val_batch, model)
        
        baseline = Optimizer(n_trials=args.trials, max_depth=args.max_depth, max_ccp_alpha=args.max_ccp_alpha, lmb=args.lmb).optimize(train_batch, args.layers + 1, logger=None, progress_bar=False)
        baseline_val_acc = baseline.accuracy(val_batch)

        expl.save_image('./figures/' + args.dataset + '_baseline_' + f"-{baseline_val_acc:.0%}")
        logger.log_image('baseline', images=['./figures/' + args.dataset + '_baseline_' + f"-{baseline_val_acc:.0%}.png"])                
        
        logger.experiment.finish()
        
        api = wandb.Api()
        current_run = api.run(f"lexpk/gnnexplain/{logger.version}")
        with torch.no_grad():
            model_predicition = model(val_batch).argmax(-1)
            current_run.summary['gcn_val_acc'] = model_predicition.eq(val_batch.y).float().mean().item()
        current_run.summary['gtree_val_acc'] = val_acc
        current_run.summary['train_fidelity'] = train_fidelity
        current_run.summary['val_fidelity'] = val_fidelity
        current_run.summary['baseline_val_acc'] = baseline_val_acc
        current_run.summary.update()
        
    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    Parallel(n_jobs=args.kfold)(delayed(run)(i, start_time, i%4) for i in range(args.kfold))