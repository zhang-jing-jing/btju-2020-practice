export CUDA_VISIBLE_DEVICES=3,5
python esim_main.py \
--learning_rate 5e-3 \
--train_or_test train \
--param_path './dataset/train_param3.bin'