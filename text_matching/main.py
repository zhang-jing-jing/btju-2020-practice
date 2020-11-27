import torch
import torch.nn as nn
import os
import pdb
import re # 正则
from mxnet import gluon

def read_snli(file_path):
    """
    读取数据
    """
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    with open(file_path) as f_file:
        rows = [row.split('\t') for row in f_file.readlines()[1:]]
        premises = [extract_text(row[1]) for row in rows if row[0] in label_set] # 前提
        hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set] # 假设
        labels = [label_set[row[0]] for row in rows if row[0] in label_set] #标签
    return premises, hypotheses, labels

label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
train_feature1 , train_feature2, train_label  = read_snli('./dataset/snli_1.0_train.txt')
dev_feature1 , dev_feature2, dev_label  = read_snli('./dataset/snli_1.0_dev.txt')
test_feature1 , test_feature2, test_label  = read_snli('./dataset/snli_1.0_test.txt')

print(train_feature1[0])

