# Explaining GNNs via iterated decision trees.

This is the code accompanying the paper

```
@article{ldgnn,
  author  = {Alexander Pluska and Pascal Welke and Thomas G\"artner and Sagar Malhotra},
  title   = {Logical Distillation of Graph Neural Networks},
  year    = {2024},
  comment = {under review},
}
```

## Setup

We recommed using a conda or virtual environment with pip installed. The required packages can then be installed by calling
```bash
pip install .
```
in the project directory. This will likely install a CPU version of pytorch. In order to use GPU acceleration, please consult the [PyTorch documentation](https://pytorch.org/get-started/locally/) on how to install the correct version for your system.

In order to run the experiments in evaluate.sh/evaluate.py you also need to [set up wandb](https://docs.wandb.ai/quickstart).

## Running the experiments

For convenice, we provide a jupyter notebook `example.ipynb` that demonstrates training a GCN and an IDT on the AIDS dataset with a single random split. It should run fine on most devices and configurations.

The experiments in the paper are performed using parallel 10-fold cross-validation and can be run via
```bash
.\evaluate.sh
```
They are configured for a node with 4 GPUs. If you have fewer GPUs, you can adjust the `--devices` argument in the `evaluate.sh` script. CPU is not supported.