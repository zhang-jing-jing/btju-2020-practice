export CUDA_VISIBLE_DEVICES=5,6
python main.py \
--learning_rate 1e-3 \
--train_or_test train \
--param_path './dataset/train_param.bin'