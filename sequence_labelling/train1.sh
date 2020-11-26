export CUDA_VISIBLE_DEVICES=6,7
python main.py \
--learning_rate  4e-3 \
--epoch_num 100 \
--hidden_size 512 \
--train_or_test train \
--param_path './dataset/train1_param.bin'

# export CUDA_VISIBLE_DEVICES=6,7
# python main.py \
# --learning_rate  4e-3 \
# --epoch_num 100 \
# --hidden_size 512 \
# --train_or_test train \
# --param_path './dataset/train1_param.bin' 0.7700