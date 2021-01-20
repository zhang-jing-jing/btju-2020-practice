export CUDA_VISIBLE_DEVICES=0,1
python esim_main.py \
--learning_rate 3e-3 \
--train_or_test train \
--param_path './dataset/train_param2.bin'