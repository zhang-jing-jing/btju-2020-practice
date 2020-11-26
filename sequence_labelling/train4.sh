export CUDA_VISIBLE_DEVICES=1,2
python main.py \
--learning_rate  5e-3 \
--epoch_num 100 \
--hidden_size 1024 \
--train_or_test train \
--param_path './dataset/train4_param.bin'