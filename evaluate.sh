python -m run --dataset AIDS          --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation add  --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m run --dataset PROTEINS      --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-2
python -m run --dataset BZR           --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-2
python -m run --dataset EMLC0         --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m run --dataset EMLC1         --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m run --dataset EMLC2         --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m run --dataset BAMultiShapes --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
