python -m evaluate --dataset AIDS          --devices 4 --kfold 10 --max_steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation add  --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluate --dataset PROTEINS      --devices 4 --kfold 10 --max_steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-2
python -m evaluate --dataset BZR           --devices 4 --kfold 10 --max_steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-2
python -m evaluate --dataset EMLC0         --devices 4 --kfold 10 --max_steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluate --dataset EMLC1         --devices 4 --kfold 10 --max_steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluate --dataset EMLC2         --devices 4 --kfold 10 --max_steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluate --dataset BAMultiShapes --devices 4 --kfold 10 --max_steps 1000 --layers 8 --dim 128 --lr 1e-4 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --sample_size 4000 --ccp_alpha 1e-3
