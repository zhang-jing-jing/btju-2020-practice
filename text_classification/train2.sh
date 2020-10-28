
export CUDA_VISIBLE_DEVICES=6,7
python main.py \
--learning_rate 1e-2 \
--hidden_size 512 \
--train_or_test train \
--param_path './dataset/train2_param.bin'
