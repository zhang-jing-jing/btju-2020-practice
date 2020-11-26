export CUDA_VISIBLE_DEVICES=5,6,7
python main.py \
--learning_rate 1e-3 \
--train_or_test train \
--param_path './dataset/train_param.bin'

#f1 0.7692