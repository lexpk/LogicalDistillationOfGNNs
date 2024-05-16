python -m evaluateTU --dataset AIDS --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation add --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluateTU --dataset PROTEINS --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-2
python -m evaluateTU --dataset BZR --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-2
python -m evaluateEMLC --formula_index 0 --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluateEMLC --formula_index 1 --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluateEMLC --formula_index 2 --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
python -m evaluateBAMultiShape --kfold 10 --steps 1000 --layers 8 --dim 128 --lr 1e-5 --activation ReLU --aggregation mean --width 8 --layer_depth 2 --ccp_alpha 1e-3
