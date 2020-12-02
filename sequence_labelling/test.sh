export CUDA_VISIBLE_DEVICES=4,5,7
python main.py \
--learning_rate  5e-3 \
--epoch_num 150 \
--hidden_size 256 \
--train_or_test test \
--test_path './conll03.test' \
--test_result_path './result2.txt' \
--param_path './param_bin/train9_param.bin'

# 0.7752