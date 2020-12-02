 export CUDA_VISIBLE_DEVICES=4,5,7
 python main.py \
 --learning_rate  5e-3 \
 --epoch_num 150 \
 --hidden_size 256 \
 --train_or_test train \
 --param_path './train8_param.bin'
