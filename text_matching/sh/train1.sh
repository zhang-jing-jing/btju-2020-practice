export CUDA_VISIBLE_DEVICES=0,2
python esim_main.py \
--learning_rate 3e-3 \
--train_or_test train \
--hidden_size 512 \
--param_path './dataset/train1_param.bin'