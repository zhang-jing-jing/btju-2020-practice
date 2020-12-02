import torch
import torch.nn as nn
import os
import pdb
import re # 正则
from gensim.models import KeyedVectors
import argparse
# 参数列表
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=17000)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--cuda', type=str, default='cuda:5')
parser.add_argument('--steps_per_eval', type=int, default=20)
parser.add_argument('--steps_per_log', type=int, default=10)
parser.add_argument('--param_path',type=str, default='./dataset/param.bin')
parser.add_argument('--test_path',type=str, default='./dataset/test/seq.in')
parser.add_argument('--test_result_path',type=str, default='./result.txt')
parser.add_argument('--train_or_test', type=str, choices=('train', 'test'), default='train')
args = parser.parse_args()

def read_snli(file_path):
    """
    读取数据function
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
    return premises, hypotheses, labels # 前提，假设，标签

def get_data(feature1, feature2, label):
    """
    :param feature1:前提
    :param feature2:假设
    :param label:标签
    :return:
    """
    data_pair = []
    for i in range(len(feature1)):
        temp = []
        temp.append(feature1[i])
        temp.append(feature2[i])
        temp.append(label[i])
        data_pair.append(temp)
    return data_pair

def collate_fn(data_pair):
    data_pair.sort(key=lambda data: len(data[0]), reverse=True)                          #倒序排序
    feature1, feature2 , label = [], [], []
    for data in data_pair:
        feature1.append(data[0])
        feature2.append(data[1])
        label.append(data[2])
    # data_length = [len(data[0]) for data in sample_data]                                   #取出所有data的长度
    feature1 = torch.nn.utils.rnn.pad_sequence(feature1, batch_first=True, padding_value=padding_value)
    feature2 = torch.nn.utils.rnn.pad_sequence(feature2, batch_first=True, padding_value=padding_value)
    return feature1, feature2, label

###################################加载数据###################################
label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2} # 蕴含，矛盾，中立
train_feature1 , train_feature2, train_label  = read_snli('./dataset/snli_1.0_train.txt')
dev_feature1 , dev_feature2, dev_label  = read_snli('./dataset/snli_1.0_dev.txt')
test_feature1 , test_feature2, test_label  = read_snli('./dataset/snli_1.0_test.txt')

###################################加载词向量##################################
word2vec_path = "../wvmodel/word2vec.txt"
wvmodel = KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
print('loading word2vec....')
# word -> id 映射表
word2id = dict(zip(wvmodel.index2word,range(len(wvmodel.index2word))))
# id -> word 映射表
id2word = {idx:word for idx,word in enumerate(wvmodel.index2word)}
word2id['[UNK]'] = len(word2id)
word2id['[PAD]'] = len(word2id)
unk = word2id['[UNK]']              #UNK:低频词
padding_value = word2id['[PAD]']    #PAD:填充词
# ###################################数据处理###################################

train_textline1 = [line.strip().lower().split(' ') for line in train_feature1]
train_textline2 = [line.strip().lower().split(' ') for line in train_feature2]
dev_textline1 = [line.strip().lower().split(' ') for line in dev_feature1]
dev_textline2 = [line.strip().lower().split(' ')for line in dev_feature2]
test_textline1 = [line.strip().lower().split(' ') for line in test_feature1]
test_textline2 = [line.strip().lower().split(' ') for line in test_feature2]

train_textline1 = [[word2id[word] if word in word2id else padding_value for word in line] for line in train_textline1]
train_textline2 = [[word2id[word] if word in word2id else padding_value for word in line] for line in train_textline2]

dev_textline1 = [[word2id[word] if word in word2id else padding_value for word in line] for line in dev_textline1]
dev_textline2 = [[word2id[word] if word in word2id else padding_value for word in line] for line in dev_textline2]

test_textline1 = [[word2id[word] if word in word2id else padding_value for word in line] for line in test_textline1]
test_textline2 = [[word2id[word] if word in word2id else padding_value for word in line] for line in test_textline2]

train_textline1 = [torch.tensor(line) for line in train_textline1]
train_textline2 = [torch.tensor(line) for line in train_textline2]

train_pre_data = get_data(train_textline1, train_textline2, train_label)
dev_pre_data = get_data(dev_textline1, dev_textline2, dev_label)
test_pre_data = get_data(test_textline1, test_textline2, test_label)

train_loader = torch.utils.data.DataLoader(dataset=train_pre_data,
                                           batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
