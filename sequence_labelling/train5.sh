export CUDA_VISIBLE_DEVICES=3,4
python main.py \
--learning_rate  6e-3 \
--epoch_num 100 \
--hidden_size 256 \
--train_or_test train \
--param_path './dataset/train5_param.bin'

# f1 0.7685