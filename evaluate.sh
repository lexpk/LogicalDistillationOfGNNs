python -m evaluateTU --dataset AIDS --steps 1000 --layers 3 --dim 64 --lr 1e-4 --dropout 0.0 --activation ReLU --batch_size 2000
python -m evaluateTU --dataset AIDS --steps 5000 --layers 3 --dim 64 --lr 0.0001 --dropout 0.5
python -m evaluateTU --dataset BZR
python -m evaluateTU --dataset BZR_MD
python -m evaluateTU --dataset MUTAG
python -m evaluateTU --dataset Mutagenicity
python -m evaluateTU --dataset ENZYMES
python -m evaluateTU --dataset PROTEINS