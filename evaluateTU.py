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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
import wandb
from joblib import Parallel, delayed

from gnnexplain.model.gtree import Explainer, _get_values
from gnnexplain.nn.gnn import GNN

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
        torch.set_float32_matmul_precision('high')
        torch.manual_seed(seed)
        
        dataset = datasets.TUDataset(root='data', name=args.dataset)
        n_val = len(dataset) // args.kfold
        permutation = torch.randperm(len(dataset))

        val_data = dataset[permutation[iteration * n_val : (iteration + 1) * n_val]]
        train_data = dataset[torch.concat([permutation[:iteration * n_val], permutation[(iteration + 1) * n_val:]])]

        val_loader = DataLoader(val_data, batch_size=min(len(val_data), args.batch_size), shuffle=False)
        train_loader = DataLoader(train_data, batch_size=min(len(train_data), args.batch_size), shuffle=True)
        
        num_classes = train_data.y.max().item() + 1
        bincount = torch.bincount(train_data.y, minlength=num_classes)
        weight = len(train_data) / (num_classes * bincount.float())
        
        os.environ["WANDB_SILENT"] = "true"
        logger = WandbLogger(project="gnnexplain", group=f'{args.dataset}_{args.activation}_{args.kfold}fold_{args.layers}layers_{args.dim}dim_{start_time}')

        GCN = GNN(dataset.num_features, dataset.num_classes, layers=args.layers, dim=args.dim, activation=args.activation, conv="GCN", aggr=args.aggregation, lr=args.lr, weight=weight)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
        trainer = Trainer(
            max_steps=args.steps,
            logger=logger,
            devices=[device],
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1
        )
        trainer.fit(GCN, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.save_checkpoint(f'./checkpoints/{args.dataset}_{args.activation}_{logger.version}.ckpt')
        GCN.eval()

        GIN = GNN(dataset.num_features, dataset.num_classes, layers=args.layers, dim=args.dim, activation=args.activation, conv="GIN", aggr=args.aggregation, lr=args.lr, weight=weight)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
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
        trainer.save_checkpoint(f'./checkpoints/{args.dataset}_{args.activation}_{logger.version}.ckpt')
        GIN.eval()
        
        with torch.no_grad():
            gcn_prediction = GCN(val_batch).argmax(-1)
            gin_prediction = GIN(val_batch).argmax(-1)

        train_batch = Batch.from_data_list(train_data)
        val_batch = Batch.from_data_list(val_data)
        sample_weight = weight[train_batch.y]

        
        def run_idt(values, y, sample_weight):
            explainer = Explainer(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha).fit(
                train_batch, values, y=y, sample_weight=sample_weight)
            explainer_prediction = explainer.predict(val_batch)
            explainer.prune()
            explainer.save_image(f'./figures/{args.dataset}_{args.activation}_{logger.version}')
            return (
                explainer_prediction.eq(val_batch.y).float().mean().item(),
                f1_score(val_batch.y.cpu(), explainer_prediction.cpu(), average='macro'),
                explainer_prediction.eq(gcn_prediction).float().mean().item(),
                explainer_prediction.eq(gin_prediction).float().mean().item()
            )

        idt_gcn_val_acc, idt_gcn_f1_macro, idt_gcn_gcn_fidelity, idt_gcn_gin_fidelity = run_idt(_get_values(train_batch, GCN), GCN(train_batch).argmax(-1), sample_weight)
        idt_gcn_true_val_acc, idt_gcn_true_f1_macro, idt_gcn_true_gcn_fidelity, idt_gcn_true_gin_fidelity = run_idt(_get_values(train_batch, GCN), train_batch.y, sample_weight)
        idt_gin_val_acc, idt_gin_f1_macro, idt_gin_gcn_fidelity, idt_gin_gin_fidelity = run_idt(_get_values(train_batch, GIN), GIN(train_batch).argmax(-1), sample_weight)
        idt_gin_true_val_acc, idt_gin_true_f1_macro, idt_gin_true_gcn_fidelity, idt_gin_true_gin_fidelity = run_idt(_get_values(train_batch, GIN), train_batch.y, sample_weight)
        idt_true_val_acc, idt_true_f1_macro, idt_true_gcn_fidelity, idt_true_gin_fidelity = run_idt(_get_values(train_batch, args.layers), train_batch.y, sample_weight)
        logger.experiment.finish()
        
        
    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    Parallel(n_jobs=args.kfold)(delayed(run)(i, start_time, i%4) for i in range(args.kfold))
