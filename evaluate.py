from argparse import ArgumentParser
from joblib import Parallel, delayed
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
from pytorch_lightning.loggers import WandbLogger
import signal
from sklearn.metrics import f1_score
import sys
import time
import torch

from idt.data import data
from idt.idt import IDT, get_activations
from idt.gnn import GNN


def run_cv(args, run_id):
    Parallel(n_jobs=args.kfold)(delayed(run_split)(args, cv_split, run_id=run_id, device=cv_split%args.devices) for cv_split in range(args.kfold))


def run_split(args, cv_split, run_id, device=0):
    torch.set_float32_matmul_precision('high')

    num_features, num_classes, train_loader, val_loader, train_val_batch, test_batch = data(args.dataset, args.kfold, cv_split, seed=42)
    
    bincount = torch.bincount(train_val_batch.y, minlength=2)
    weight = len(train_val_batch) / (2 * bincount.float())
    
    os.environ["WANDB_SILENT"] = "true"
    logger = WandbLogger(project="gnnexplain", group=f'{args.dataset}_{run_id}')

    GCN = GNN(num_features, num_classes, layers=args.layers, dim=args.dim, activation=args.activation, conv="GCN", pool=args.pooling, lr=args.lr, weight=weight)
    early_stop_callback = EarlyStopping(monitor="GCN_val_loss", patience=10, mode="min")
    trainer = Trainer(
        max_steps=args.max_steps,
        callbacks=[early_stop_callback],
        logger=logger,
        devices=[device],
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=1
    )
    trainer.fit(GCN, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f'./checkpoints/{args.dataset}_GCN_{logger.version}.ckpt')
    GCN.eval()

    GIN = GNN(num_features, num_classes, layers=args.layers, dim=args.dim, activation=args.activation, conv="GIN", pool=args.pooling, lr=args.lr, weight=weight)
    early_stop_callback = EarlyStopping(monitor="GIN_val_loss", patience=10, mode="min")
    trainer = Trainer(
        max_steps=args.max_steps,
        callbacks=[early_stop_callback],
        logger=logger,
        devices=[device],
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=1
    )
    trainer.fit(GIN, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f'./checkpoints/{args.dataset}_GIN_{logger.version}.ckpt')
    GIN.eval()
    
    with torch.no_grad():
        gcn_prediction = GCN(test_batch).argmax(-1).detach().numpy()
        gin_prediction = GIN(test_batch).argmax(-1).detach().numpy()

    logger.experiment.summary['gcn_test_acc'] = (gcn_prediction == test_batch.y.detach().numpy()).mean()
    logger.experiment.summary['gcn_f1_macro'] = f1_score(test_batch.y.detach().numpy(), gcn_prediction, average='macro')
    logger.experiment.summary['gcn_gcn_fidelity'] = (gcn_prediction == gcn_prediction).mean()
    logger.experiment.summary['gin_gcn_fidelity'] = (gcn_prediction == gin_prediction).mean()
    logger.experiment.summary['gin_test_acc'] = (gin_prediction == test_batch.y.detach().numpy()).mean()
    logger.experiment.summary['gin_f1_macro'] = f1_score(test_batch.y.detach().numpy(), gin_prediction, average='macro')
    logger.experiment.summary['gcn_gin_fidelity'] = (gin_prediction == gcn_prediction).mean()
    logger.experiment.summary['gin_gin_fidelity'] = (gin_prediction == gin_prediction).mean()

    sample_weight = weight[train_val_batch.y]
    
    def run_idt(values, y, sample_weight):
        idt = IDT(width=args.width, sample_size=args.sample_size, layer_depth=args.layer_depth, max_depth=args.max_depth, ccp_alpha=args.ccp_alpha).fit(
            train_val_batch, values, y=y, sample_weight=sample_weight)
        explainer_prediction = idt.predict(test_batch)
        test_accuracy = (explainer_prediction == test_batch.y.detach().numpy()).mean()
        idt.prune()
        idt.save_image(f'./figures/{args.dataset}_{args.activation}_{logger.version}_{test_accuracy}.png')
        return (
            test_accuracy,
            f1_score(test_batch.y.detach().numpy(), explainer_prediction, average='macro'),
            (explainer_prediction == gcn_prediction).mean(),
            (explainer_prediction == gin_prediction).mean()
        )
    
    idt_gcn_test_acc, idt_gcn_f1_macro, idt_gcn_gcn_fidelity, idt_gcn_gin_fidelity = \
        run_idt(get_activations(train_val_batch, GCN), GCN(train_val_batch).argmax(-1), sample_weight)
    logger.experiment.summary['idt_gcn_test_acc'] = idt_gcn_test_acc
    logger.experiment.summary['idt_gcn_f1_macro'] = idt_gcn_f1_macro
    logger.experiment.summary['idt_gcn_gcn_fidelity'] = idt_gcn_gcn_fidelity
    logger.experiment.summary['idt_gcn_gin_fidelity'] = idt_gcn_gin_fidelity
    
    idt_gcn_true_test_acc, idt_gcn_true_f1_macro, idt_gcn_true_gcn_fidelity, idt_gcn_true_gin_fidelity = \
        run_idt(get_activations(train_val_batch, GCN), train_val_batch.y.detach().numpy(), sample_weight)
    logger.experiment.summary['idt_gcn_true_test_acc'] = idt_gcn_true_test_acc
    logger.experiment.summary['idt_gcn_true_f1_macro'] = idt_gcn_true_f1_macro
    logger.experiment.summary['idt_gcn_true_gcn_fidelity'] = idt_gcn_true_gcn_fidelity
    logger.experiment.summary['idt_gcn_true_gin_fidelity'] = idt_gcn_true_gin_fidelity
    
    idt_gin_test_acc, idt_gin_f1_macro, idt_gin_gcn_fidelity, idt_gin_gin_fidelity = \
        run_idt(get_activations(train_val_batch, GIN), GIN(train_val_batch).argmax(-1), sample_weight)
    logger.experiment.summary['idt_gin_test_acc'] = idt_gin_test_acc
    logger.experiment.summary['idt_gin_f1_macro'] = idt_gin_f1_macro
    logger.experiment.summary['idt_gin_gcn_fidelity'] = idt_gin_gcn_fidelity
    logger.experiment.summary['idt_gin_gin_fidelity'] = idt_gin_gin_fidelity
    
    idt_gin_true_test_acc, idt_gin_true_f1_macro, idt_gin_true_gcn_fidelity, idt_gin_true_gin_fidelity = \
        run_idt(get_activations(train_val_batch, GIN), train_val_batch.y, sample_weight)
    logger.experiment.summary['idt_gin_true_test_acc'] = idt_gin_true_test_acc
    logger.experiment.summary['idt_gin_true_f1_macro'] = idt_gin_true_f1_macro
    logger.experiment.summary['idt_gin_true_gcn_fidelity'] = idt_gin_true_gcn_fidelity
    logger.experiment.summary['idt_gin_true_gin_fidelity'] = idt_gin_true_gin_fidelity
    
    idt_true_test_acc, idt_true_f1_macro, idt_true_gcn_fidelity, idt_true_gin_fidelity = \
        run_idt(get_activations(train_val_batch, args.layers), train_val_batch.y.detach().numpy(), sample_weight)
    logger.experiment.summary['idt_true_test_acc'] = idt_true_test_acc
    logger.experiment.summary['idt_true_f1_macro'] = idt_true_f1_macro
    logger.experiment.summary['idt_true_gcn_fidelity'] = idt_true_gcn_fidelity
    logger.experiment.summary['idt_true_gin_fidelity'] = idt_true_gin_fidelity
    logger.experiment.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='EMLC0', help='Name of the dataset')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation Function', choices=['ReLU', 'Tanh', 'Sigmoid'])
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling Function', choices=['mean', 'add'])
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--kfold', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of node embeddings')
    parser.add_argument('--max_steps', type=int, default=1000, help='Upper bound for the number of training steps')
    parser.add_argument('--width', type=int, default=10, help='Number of decision trees per layer')
    parser.add_argument('--sample_size', type=int, default=None, help='Size of subsamples to train decision trees on')
    parser.add_argument('--layer_depth', type=int, default=2, help='Depth of iterated decision trees')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of final tree')
    parser.add_argument('--ccp_alpha', type=float, default=1e-3, help='ccp_alpha of final tree')
    parser.add_argument('--devices', type=int, default=4, help='Number of devices')

    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parser.parse_args()
    run_cv(args, time.time())
