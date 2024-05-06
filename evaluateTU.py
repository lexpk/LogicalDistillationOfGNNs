import os
import time
import optuna
import signal
import sys
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.data import Batch
from lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
from joblib import Parallel, delayed

from gnnexplain.model.gtree import Explainer, _get_values
from gnnexplain.nn.gnn import GCN

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AIDS', help='Name of the dataset')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation Function', choices=['ReLU', 'Tanh', 'Sigmoid'])
    parser.add_argument('--aggregation', type=str, default='mean', help='Aggregation Function', choices=['mean', 'add'])
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--kfold', type=int, default=8, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--dim', type=int, default=64, help='Dimension of node embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--width', type=int, default=10, help='Number of decision trees per layer')
    parser.add_argument('--sample_size', type=int, default=100, help='Size of subsamples to train decision trees on')
    parser.add_argument('--layer_depth', type=int, default=2, help='Depth of iterated decision trees')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of final tree')
    parser.add_argument('--ccp_alpha', type=float, default=1e-3, help='ccp_alpha of final tree')
    

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

        val_loader = DataLoader(val_data, batch_size=min(len(val_data), args.batch_size), shuffle=False)
        train_loader = DataLoader(train_data, batch_size=min(len(train_data), args.batch_size), shuffle=True)
        
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
        
        model = GCN(dataset.num_features, dataset.num_classes, layers=args.layers, dim=args.dim, activation=args.activation, aggr=args.aggregation, lr=args.lr, dropout=args.dropout, weight=weight)
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
        model.eval()

        train_batch = Batch.from_data_list(train_data)
        val_batch = Batch.from_data_list(val_data)
        sample_weight = weight[train_batch.y]

        idt0 = Explainer(width=25, sample_size=100, layer_depth=2, max_depth=10, ccp_alpha=1e-3).fit(
            train_batch, _get_values(train_batch, model), y=model(train_batch).argmax(-1), sample_weight=sample_weight)
        idt0.prune()
        idt0_val_acc = idt0.accuracy(val_batch)
        idt0_f1_macro = idt0.f1_score(val_batch, average='macro')
        idt0_f1_micro = idt0.f1_score(val_batch, average='micro')
        idt0_fidelity = idt0.fidelity(val_batch, model)
        idt0.save_image('./figures/idt0_' + args.dataset + f'-{idt0_val_acc:.0%}.png')
        
        idt1 = Explainer(width=10, sample_size=100, layer_depth=2, max_depth=10, ccp_alpha=1e-3).fit(
            train_batch, _get_values(train_batch, model), y=train_batch.y, sample_weight=sample_weight)
        idt1.prune()
        idt1_val_acc = idt1.accuracy(val_batch)
        idt1_f1_macro = idt1.f1_score(val_batch, average='macro')
        idt1_f1_micro = idt1.f1_score(val_batch, average='micro')
        idt1_fidelity = idt1.fidelity(val_batch, model)
        idt1.save_image('./figures/idt1_' + args.dataset + f'-{idt1_val_acc:.0%}.png')            

        idt2 = Explainer(width=10, sample_size=100, layer_depth=2, max_depth=10, ccp_alpha=1e-3).fit(
            train_batch, _get_values(train_batch, args.layers), y=train_batch.y, sample_weight=sample_weight)
        idt2.prune()
        idt2_val_acc = idt2.accuracy(val_batch)
        idt2_f1_macro = idt2.f1_score(val_batch, average='macro')
        idt2_f1_micro = idt2.f1_score(val_batch, average='micro')
        idt2_fidelity = idt2.fidelity(val_batch, model)
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