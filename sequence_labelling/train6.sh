export CUDA_VISIBLE_DEVICES=3,4
python main.py \
--learning_rate  5e-3 \
--epoch_num 150 \
--hidden_size 128 \
--train_or_test train \
--param_path './dataset/train6_param.bin'

