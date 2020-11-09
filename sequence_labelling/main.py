import os
import torch
import argparse
from collections import defaultdict,Counter
import torchtext.vocab as Vectors
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 参数列表
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=18000)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--max_len', type=int, default=500)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--output_dim',type=int,default=2)
parser.add_argument('--kernel_size',type=list,default=[2,3,4,5])
parser.add_argument('--kernel_num',type=int,default=256)#每种卷积和的大小
parser.add_argument('--steps_per_eval', type=int, default=20)
parser.add_argument('--steps_per_log', type=int, default=10)
parser.add_argument('--param_path',type=str, default='./dataset/param.bin')
parser.add_argument('--test_path',type=str, default='./dataset/test2.txt')
parser.add_argument('--train_or_test', type=str, choices=('train', 'test'), default='train')
args = parser.parse_args()

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

def get_vocab(train_feature):
    if os.path.exists('./dataset/vocab.txt'):
        with open('./dataset/vocab.txt',"r",encoding='utf-8') as fvocab:
            vocab_words = [line for line in fvocab]
    else:
        train_word = []
        for line in train_feature:
            train_word.extend(line)
        counter = Counter(train_word)
        common_words = counter.most_common()
        vocab_words = [word[0] for word in common_words[:args.vocab_size-2]]
        vocab_words = ['[UNK]','[PAD]'] + vocab_words
        with open("./dataset/vocab.txt","w",encoding='utf-8') as fvocab:
                for word in vocab_words:
                    fvocab.write(word+'\n')
    return vocab_words

# Vectors.pretrained_aliases.keys()

# glove_dir = "./dataset/glove"

# glove = vocab.GloVe(name='6B', dim=100, cache=glove_dir)
# """
# stoi: 词到索引的字典：
# itos: 一个列表，索引到词的映射；
# vectors: 词向量
# """
# print("一共包含%d个词。" % len(glove.stoi))
# print(glove.stoi['beautiful'])
# print(glove.itos[3306])

word2vec_path = "./dataset/word2vec.txt"
def get_wvmodel():
    if os.path.exists(word2vec_path):
        # 使用gensim载入word2vec词向量
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
    else:
        # 已有的glove词向量
        glove_file = './dataset/glove/glove.6B.300d.txt'
        # 指定转化为word2vec格式后文件的位置
        tmp_file = word2vec_path
        #glove词向量转化为word2vec词向量的格式
        glove2word2vec(glove_file, tmp_file)
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
    return wvmodel

def get_weight(wvmodel):
    """"
    去词向量文件中查表，得到词表中单词对应的权重weight
    """
    weight = torch.zeros(args.vocab_size, args.embedding_size)
    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
    return weight



train_feature, train_label = read_file('./dataset/train/seq.in','./dataset/train/seq.out')
vaild_feature, valid_label = read_file('./dataset/valid/seq.in','./dataset/valid/seq.out')
# test_feature, test_label = read_file('./dataset/test/seq.in','./dataset/test/seq.out')

vocab_words = get_vocab(train_feature)

wvmodel = get_wvmodel()
weight = get_weight(wvmodel)
# embedding = nn.Embedding.from_pretrained(weight)
# # requires_grad指定是否在训练过程中对词向量的权重进行微调
# self.embedding.weight.requires_grad = True

