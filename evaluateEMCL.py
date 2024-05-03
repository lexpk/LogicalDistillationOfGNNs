import os
import time
import optuna
import random
import signal
from scipy.sparse import coo_matrix
import sys
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import torch
import wandb
from joblib import Parallel, delayed

from gnnexplain.model.gtree import Optimizer
from gnnexplain.nn.gnn import GNN
from gnnexplain.syntheticEMCL.datasets import generate

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--formula_index', type=int, default=1, help='Index of the to label formula')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation Function', choices=['ReLU', 'Tanh', 'Sigmoid'])
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--kfold', type=int, default=8, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dim', type=int, default=32, help='Dimension of node embeddings')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--steps', type=int, default=200, help='Number of training steps')
    parser.add_argument('--trials', type=int, default=1000, help='Number of Optuna trials')
    parser.add_argument('--max_depth', type=int, default=4, help='Maximum depth of the explanation tree')
    parser.add_argument('--max_ccp_alpha', type=float, default=0.0, help='Maximum ccp_alpha of the explanation tree')
    parser.add_argument('--lmb', type=float, default=0.01, help='Lambda for the regularization term')
    

    args = parser.parse_args()

    start_time = time.time()
    seed = 42

    def run(iteration, start_time, device=0):
        torch.set_float32_matmul_precision('medium')
        torch.manual_seed(seed)
        random.seed(seed)

        datalist = [
            generate(args.formula_index) for _ in range(1000)
        ]
        random.shuffle(datalist)
        
        train_data = datalist[iteration * 125 : (iteration + 1) * 125]
        val_data = datalist[:iteration * 125] + datalist[(iteration + 1) * 125:]
        
        val_loader = DataLoader(val_data, batch_size=875, shuffle=False, num_workers=64//args.kfold, multiprocessing_context='fork')
        train_loader = DataLoader(train_data, batch_size=125, shuffle=True, num_workers=64//args.kfold, multiprocessing_context='fork')

        model = GNN(1, 2, layers=args.layers, dim=args.dim, activation=args.activation, lr=args.lr, dropout=args.dropout)
        os.environ["WANDB_SILENT"] = "true"
        logger = WandbLogger(project="gnnexplain", group=f'EMCL{args.formula_index}_{args.activation}_{args.kfold}fold_{args.layers}layers_{args.dim}dim_{start_time}')

        trainer = Trainer(
            max_steps=args.steps,
            logger=logger,
            devices=[device],
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint(f'./checkpoints/EMCL{args.formula_index}_{args.activation}_{logger.version}.ckpt')

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        train_batch = next(iter(train_loader))
        expl = Optimizer(n_trials=args.trials, max_depth=args.max_depth, max_ccp_alpha=args.max_ccp_alpha, lmb=args.lmb).optimize(train_batch, model, logger=logger, progress_bar=False)
        val_batch = next(iter(val_loader))
        val_acc = expl.accuracy(val_batch)

        expl.save_image(f'./figures/EMCL{args.formula_index}-{val_acc:.0%}.png')
        logger.log_image('explanation', images=[f'./figures/EMCL{args.formula_index}-{val_acc:.0%}.png'])
        
        train_fidelity = expl.fidelity(train_batch, model)
        val_fidelity = expl.fidelity(val_batch, model)
        
        baseline = Optimizer(n_trials=args.trials, max_depth=args.max_depth, max_ccp_alpha=args.max_ccp_alpha, lmb=args.lmb).optimize(train_batch, args.layers + 1, logger=None, progress_bar=False)
        baseline_val_acc = baseline.accuracy(val_batch)

        expl.save_image(f'./figures/EMCL{args.formula_index}_baseline-{baseline_val_acc:.0%}.png')
        logger.log_image('baseline', images=[f'./figures/EMCL{args.formula_index}_baseline-{baseline_val_acc:.0%}.png'])                
        
        logger.experiment.finish()
        
        api = wandb.Api()
        current_run = api.run(f"lexpk/gnnexplain/{logger.version}")
        with torch.no_grad():
            current_run.summary['gcn_val_acc'] = model(val_batch).argmax(-1).eq(val_batch.y).float().mean().item()
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