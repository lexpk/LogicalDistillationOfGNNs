import os
import time
import signal
import numpy as np
from scipy.sparse import coo_matrix
import sys
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
from joblib import Parallel, delayed
import random

from gnnexplain.model.gtree import Explainer, _get_values
from gnnexplain.nn.gnn import GNN
from gnnexplain.BAMultiShapes.generate_dataset import generate

from argparse import ArgumentParser


def adj_to_edge_index(adj):
    matrix = coo_matrix(adj)
    return torch.stack([torch.tensor(matrix.row, dtype=torch.long), torch.tensor(matrix.col, dtype=torch.long)])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation Function', choices=['ReLU', 'Tanh', 'Sigmoid'])
    parser.add_argument('--aggregation', type=str, default='mean', help='Aggregation Function', choices=['mean', 'add'])
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--kfold', type=int, default=8, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of node embeddings')
    parser.add_argument('--conv', type=str, default='GCN', help='Type of convolution layer', choices=['GCN', 'GIN'])
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--trials', type=int, default=1000, help='Number of Optuna trials')
    parser.add_argument('--max_depth', type=int, default=4, help='Maximum depth of the explanation tree')
    parser.add_argument('--max_ccp_alpha', type=float, default=0.01, help='Maximum ccp_alpha of the explanation tree')
    parser.add_argument('--lmb', type=float, default=0.1, help='Lambda for the regularization term')
    

    args = parser.parse_args()

    start_time = time.time()
    seed = 42

    def run(iteration, start_time, device=0):
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(seed)
        random.seed(seed)

        datalist = [
            Data(x=torch.tensor(np.ones((adj.shape[0], 1)), dtype=torch.float), edge_index=adj_to_edge_index(adj), y=label)
            for adj, _, label in zip(*generate(1000, seed=seed))
        ]
        random.shuffle(datalist)
        
        n_val = len(datalist) // args.kfold
        
        val_data = datalist[iteration * n_val : (iteration + 1) * n_val]
        train_data = datalist[:iteration * n_val] + datalist[(iteration + 1) * n_val:]
        
        train_loader = DataLoader(train_data, batch_size=(args.kfold - 1) * n_val, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=n_val, shuffle=False)

        model = GNN(1, 2, layers=args.layers, dim=args.dim, activation=args.activation, aggr=args.aggregation, lr=args.lr, conv=args.conv)
        os.environ["WANDB_SILENT"] = "true"
        logger = WandbLogger(project="gnnexplain", group=f'BAMultiShape_{args.conv}_{args.activation}_{args.kfold}fold_{args.layers}layers_{args.dim}dim_{start_time}')

        trainer = Trainer(
            max_steps=args.steps,
            logger=logger,
            devices=[device],
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint(f'./checkpoints/BAMultiShape_{args.activation}_{logger.version}.ckpt')
        model.eval()

        train_batch = Batch.from_data_list(train_data)
        val_batch = Batch.from_data_list(val_data)

        idt0 = Explainer(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha).fit(
            train_batch, _get_values(train_batch, model), y=model(train_batch).argmax(-1))
        idt0_val_acc = idt0.accuracy(val_batch)
        idt0_f1_macro = idt0.f1_score(val_batch, average='macro')
        idt0_f1_micro = idt0.f1_score(val_batch, average='micro')
        idt0_fidelity = idt0.fidelity(val_batch, model)
        idt0.prune()
        idt0.save_image('./figures/idt0_' + args.dataset + f'-{idt0_val_acc:.0%}.png')
        
        idt1 = Explainer(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha).fit(
            train_batch, _get_values(train_batch, model), y=train_batch.y)
        idt1_val_acc = idt1.accuracy(val_batch)
        idt1_f1_macro = idt1.f1_score(val_batch, average='macro')
        idt1_f1_micro = idt1.f1_score(val_batch, average='micro')
        idt1_fidelity = idt1.fidelity(val_batch, model)
        idt1.prune()
        idt1.save_image('./figures/idt1_' + args.dataset + f'-{idt1_val_acc:.0%}.png')            

        idt2 = Explainer(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha).fit(
            train_batch, _get_values(train_batch, args.layers), y=train_batch.y)
        idt2_val_acc = idt2.accuracy(val_batch)
        idt2_f1_macro = idt2.f1_score(val_batch, average='macro')
        idt2_f1_micro = idt2.f1_score(val_batch, average='micro')
        idt2_fidelity = idt2.fidelity(val_batch, model)
        idt2.prune()
        idt2.save_image('./figures/idt2_' + args.dataset + f'-{idt2_val_acc:.0%}.png')
        
        logger.experiment.finish()
        
        api = wandb.Api()
        current_run = api.run(f"lexpk/gnnexplain/{logger.version}")
        with torch.no_grad():
            model_predicition = model(val_batch).argmax(-1)
            current_run.summary['gcn_val_acc'] = model_predicition.eq(val_batch.y).float().mean().item()
            current_run.summary['gcn_f1_macro'] = f1_score(val_batch.y.cpu(), model_predicition.cpu(), average='macro')
            current_run.summary['gcn_f1_micro'] = f1_score(val_batch.y.cpu(), model_predicition.cpu(), average='micro')
        current_run.summary['idt0_val_acc'] = idt0_val_acc
        current_run.summary['idt0_f1_macro'] = idt0_f1_macro
        current_run.summary['idt0_f1_micro'] = idt0_f1_micro
        current_run.summary['idt0_fidelity'] = idt0_fidelity
        current_run.summary['idt1_val_acc'] = idt1_val_acc
        current_run.summary['idt1_f1_macro'] = idt1_f1_macro
        current_run.summary['idt1_f1_micro'] = idt1_f1_micro
        current_run.summary['idt1_fidelity'] = idt1_fidelity
        current_run.summary['idt2_val_acc'] = idt2_val_acc
        current_run.summary['idt2_f1_macro'] = idt2_f1_macro
        current_run.summary['idt2_f1_micro'] = idt2_f1_micro
        current_run.summary['idt2_fidelity'] = idt2_fidelity
        current_run.summary.update()
        
    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    Parallel(n_jobs=args.kfold)(delayed(run)(i, start_time, i%4) for i in range(args.kfold))
