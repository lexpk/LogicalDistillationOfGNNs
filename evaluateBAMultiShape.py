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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
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
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--width', type=int, default=10, help='Number of decision trees per layer')
    parser.add_argument('--sample_size', type=int, default=None, help='Size of subsamples to train decision trees on')
    parser.add_argument('--layer_depth', type=int, default=2, help='Depth of iterated decision trees')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of final tree')
    parser.add_argument('--ccp_alpha', type=float, default=1e-3, help='ccp_alpha of final tree')
    

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

        train_batch = Batch.from_data_list(train_data)
        val_batch = Batch.from_data_list(val_data)
        
        bincount = torch.bincount(train_batch.y, minlength=2)
        weight = len(train_data) / (2 * bincount.float())
        
        os.environ["WANDB_SILENT"] = "true"
        logger = WandbLogger(project="gnnexplain", group=f'BAMulti_{args.activation}_{args.kfold}fold_{args.layers}layers_{args.dim}dim_{start_time}')

        GCN = GNN(1, 2, layers=args.layers, dim=args.dim, activation=args.activation, conv="GCN", aggr=args.aggregation, lr=args.lr, weight=weight)
        early_stop_callback = EarlyStopping(monitor="GCN_val_loss", patience=10, mode="min")
        trainer = Trainer(
            max_steps=args.steps,
            callbacks=[early_stop_callback],
            logger=logger,
            devices=[device],
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1
        )
        trainer.fit(GCN, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint(f'./checkpoints/BAMulti_GCN_{logger.version}.ckpt')
        GCN.eval()

        GIN = GNN(1, 2, layers=args.layers, dim=args.dim, activation=args.activation, conv="GIN", aggr=args.aggregation, lr=args.lr, weight=weight)
        early_stop_callback = EarlyStopping(monitor="GIN_val_loss", patience=10, mode="min")
        trainer = Trainer(
            max_steps=args.steps,
            callbacks=[early_stop_callback],
            logger=logger,
            devices=[device],
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1
        )
        trainer.fit(GIN, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint(f'./checkpoints/BAMulti_GIN_{logger.version}.ckpt')
        GIN.eval()
        
        with torch.no_grad():
            gcn_prediction = GCN(val_batch).argmax(-1).detach().numpy()
            gin_prediction = GIN(val_batch).argmax(-1).detach().numpy()

        logger.experiment.summary['gcn_val_acc'] = (gcn_prediction == val_batch.y.detach().numpy()).mean()
        logger.experiment.summary['gcn_f1_macro'] = f1_score(val_batch.y.detach().numpy(), gcn_prediction, average='macro')
        logger.experiment.summary['gcn_gcn_fidelity'] = (gcn_prediction == gcn_prediction).mean()
        logger.experiment.summary['gin_gcn_fidelity'] = (gcn_prediction == gin_prediction).mean()
        logger.experiment.summary['gin_val_acc'] = (gin_prediction == val_batch.y.detach().numpy()).mean()
        logger.experiment.summary['gin_f1_macro'] = f1_score(val_batch.y.detach().numpy(), gin_prediction, average='macro')
        logger.experiment.summary['gcn_gin_fidelity'] = (gin_prediction == gcn_prediction).mean()
        logger.experiment.summary['gin_gin_fidelity'] = (gin_prediction == gin_prediction).mean()

        sample_weight = weight[train_batch.y]
        
        def run_idt(values, y, sample_weight):
            explainer = Explainer(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha).fit(
                train_batch, values, y=y, sample_weight=sample_weight)
            explainer_prediction = explainer.predict(val_batch)
            acc = (explainer_prediction == val_batch.y.detach().numpy()).mean()
            explainer.prune()
            explainer.save_image(f'./figures/BAMulti_{args.activation}_{logger.version}_{acc}.png')
            return (
                acc,
                f1_score(val_batch.y.detach().numpy(), explainer_prediction, average='macro'),
                (explainer_prediction == gcn_prediction).mean(),
                (explainer_prediction == gin_prediction).mean()
            )
        
        idt_gcn_val_acc, idt_gcn_f1_macro, idt_gcn_gcn_fidelity, idt_gcn_gin_fidelity = run_idt(_get_values(train_batch, GCN), GCN(train_batch).argmax(-1), sample_weight)
        logger.experiment.summary['idt_gcn_val_acc'] = idt_gcn_val_acc
        logger.experiment.summary['idt_gcn_f1_macro'] = idt_gcn_f1_macro
        logger.experiment.summary['idt_gcn_gcn_fidelity'] = idt_gcn_gcn_fidelity
        logger.experiment.summary['idt_gcn_gin_fidelity'] = idt_gcn_gin_fidelity
        
        idt_gcn_true_val_acc, idt_gcn_true_f1_macro, idt_gcn_true_gcn_fidelity, idt_gcn_true_gin_fidelity = run_idt(_get_values(train_batch, GCN), train_batch.y.detach().numpy(), sample_weight)
        logger.experiment.summary['idt_gcn_true_val_acc'] = idt_gcn_true_val_acc
        logger.experiment.summary['idt_gcn_true_f1_macro'] = idt_gcn_true_f1_macro
        logger.experiment.summary['idt_gcn_true_gcn_fidelity'] = idt_gcn_true_gcn_fidelity
        logger.experiment.summary['idt_gcn_true_gin_fidelity'] = idt_gcn_true_gin_fidelity
        
        idt_gin_val_acc, idt_gin_f1_macro, idt_gin_gcn_fidelity, idt_gin_gin_fidelity = run_idt(_get_values(train_batch, GIN), GIN(train_batch).argmax(-1), sample_weight)
        logger.experiment.summary['idt_gin_val_acc'] = idt_gin_val_acc
        logger.experiment.summary['idt_gin_f1_macro'] = idt_gin_f1_macro
        logger.experiment.summary['idt_gin_gcn_fidelity'] = idt_gin_gcn_fidelity
        logger.experiment.summary['idt_gin_gin_fidelity'] = idt_gin_gin_fidelity
        
        idt_gin_true_val_acc, idt_gin_true_f1_macro, idt_gin_true_gcn_fidelity, idt_gin_true_gin_fidelity = run_idt(_get_values(train_batch, GIN), train_batch.y, sample_weight)
        logger.experiment.summary['idt_gin_true_val_acc'] = idt_gin_true_val_acc
        logger.experiment.summary['idt_gin_true_f1_macro'] = idt_gin_true_f1_macro
        logger.experiment.summary['idt_gin_true_gcn_fidelity'] = idt_gin_true_gcn_fidelity
        logger.experiment.summary['idt_gin_true_gin_fidelity'] = idt_gin_true_gin_fidelity
        
        idt_true_val_acc, idt_true_f1_macro, idt_true_gcn_fidelity, idt_true_gin_fidelity = run_idt(_get_values(train_batch, args.layers), train_batch.y.detach().numpy(), sample_weight)
        logger.experiment.summary['idt_true_val_acc'] = idt_true_val_acc
        logger.experiment.summary['idt_true_f1_macro'] = idt_true_f1_macro
        logger.experiment.summary['idt_true_gcn_fidelity'] = idt_true_gcn_fidelity
        logger.experiment.summary['idt_true_gin_fidelity'] = idt_true_gin_fidelity
        logger.experiment.finish()
        
    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    Parallel(n_jobs=args.kfold)(delayed(run)(i, start_time, i%4) for i in range(args.kfold))
