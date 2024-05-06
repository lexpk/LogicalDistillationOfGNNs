python -m evaluateTU --dataset AIDS --steps 1000 --layers 4 --dim 128 --lr 1e-4 --dropout 0.0 --activation ReLU --aggregation add --batch_size 2000 --width 10 --layer_depth 2 --sample_size 100
python -m evaluateTU --dataset Mutagenicity --steps 500 --layers 3 --dim 512 --lr 1e-5 --dropout 0.0 --activation ReLU --aggregation mean --batch_size 4000 --width 100 --layer_depth 2 --sample_size 50
