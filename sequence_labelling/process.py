import os
import torch

def read_file(feature_path,label_path):
    """
    分词 feature转化为小写
    """
    feature = []
    label = []
    with open(feature_path) as seq_in:
        for line in seq_in.readlines():
            feature.append(line.strip().lower().split(' '))
    with open(label_path) as seq_out:
        for line in seq_out.readlines():
            label.append(line.strip().split(' '))
    return feature, label

train_feature, train_label = read_file('./dataset/train/seq.in','./dataset/train/seq.out')
vaild_feature, valid_label = read_file('./dataset/valid/seq.in','./dataset/valid/seq.out')
# test_feature, test_label = read_file('./dataset/test/seq.in','./dataset/test/seq.out')
