import os
import torch
import torch.nn as nn
import argparse
from collections import defaultdict,Counter
import torchtext.vocab as Vectors
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pdb 
from torch.utils.data import Dataset, DataLoader

# 参数列表
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=18000)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--steps_per_eval', type=int, default=20)
parser.add_argument('--steps_per_log', type=int, default=10)
parser.add_argument('--param_path',type=str, default='./dataset/param.bin')
parser.add_argument('--test_path',type=str, default='./dataset/test2.txt')
parser.add_argument('--train_or_test', type=str, choices=('train', 'test'), default='train')
args = parser.parse_args()

# if torch.cuda.is_available():
#     print("using cuda")
#     device = torch.device(args.cuda)


START_TAG = 'START'
STOP_TAG = 'STOP'

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
    return feature[:100], label[:100]

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
    print('what loading glove ?')
    if os.path.exists(word2vec_path):
        print('gensim loading word2vec')
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

class SentenceDataSet(Dataset):
    def __init__(self, sent, sent_label):
        self.sent = sent
        self.sent_label = sent_label

    def __getitem__(self, item):
        return torch.Tensor(self.sent[item]), torch.Tensor(self.sent_label[item])

    def __len__(self):
        return len(self.sent)

def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、以及label列表/实际长度的列表、
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    padded_sent_seq = torch.nn.utils.rnn.pad_sequence(sent_seq, batch_first=True, padding_value=padding_value)
    return padded_sent_seq, label, data_length
    

# 模型
class BiLSTM_CRF(nn.Module):
    def __init__(self,input_dim,vocab_size, tag_to_ix,embedding_vector, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)  
        self.word_embeds = nn.Embedding(input_dim, embedding_dim)
        self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.word_embeds.weight.requires_grad = False
        # self.word_embeds = nn.Embedding.from_pretrained(pretrained_weight)
        # self.word_embeds.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
                
        # 将LSTM的输出映射到标记空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移参数矩阵 转移i到j的分数
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 这两个语句强制执行了这样的约束，我们不会将其转移到开始标记，也不会将其转移到停止标记
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # 正向算法计算分块函数
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        # START_TAG 包含所有的分数
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # 遍力整个句子
        for feat in feats:
            alphas_t = []  # 保存每个时刻的一个tensor
            for next_tag in range(self.tagset_size):
                # 发射分数 不管之前是啥，它都是一样的
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # trans_score 的第i项是从i转换到next_tag的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var 的第i个条目是执行log-sum-exp之前的边(i -> next_tag)的值
                next_tag_var = forward_var + trans_score + emit_score
                # 这个tag的 前向变量是所有分数的log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        pdb.set_trace()
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        # 给出所提供的标记序列的分数
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # 在对数空间中初始化viterbi变量。
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # 第i步的forward_var存放第i-1步的viterbi变量。
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 这个步骤的backpointers
            viterbivars_t = []  # 这个步骤的viterbi变量

            for next_tag in range(self.tagset_size):
                """
                next_tag_var[i]保存上一步tag i的变量，加上从标记i转换到next_tag的分数。
                我们这里不包括发射分数，因为最大的
                不依赖于它们(我们在下面添加了它们)
                """
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            #现在加入发射分数，并将forward_var分配给我们刚才计算的viterbi变量集。
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 按照后面的来解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 把start去掉
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        pdb.set_trace()
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def train(args, train_loader, model,optim,criterion):
    """
    训练函数
    """
    loss_log = []
    global_step = 0
    best_eval_acc = 0.0
    for epoch in range(args.epoch_num):
        total_acc,total_loss,correct,sample_num= 0, 0, 0,0
        for feature, batch_labels, data_length in train_loader:
            feature = torch.tensor(feature.clone().detach(), dtype=torch.long)
            print(batch_labels)

            # batch_labels = torch.tensor(batch_labels, dtype=torch.long)

            model.train()
            optim.zero_grad()
            
            loss = model.neg_log_likelihood(feature, batch_labels)

            loss.backward()
            optim.step()
            global_step += 1
            total_loss += loss.item()
            loss_log.append(loss.item())
        # if global_step % args.steps_per_log == 0:
        print('Train {:d}| Loss:{:.5f}'.format(epoch+1 ,total_loss / len(train_loader)))
        # if global_step % args.steps_per_eval == 0:
        #     test_acc, test_loss = evaluate_accuracy(eval_loader,model, criterion)
        #     print('at train step %d, eval accuracy is %.4f, eval loss is %.4f' % (global_step, test_acc, test_loss))

        #     if test_acc > best_eval_acc:
        #         best_eval_acc = test_acc
        #         torch.save(model.state_dict(), args.param_path)
    

if args.train_or_test == "train":
    train_feature, train_label = read_file('./dataset/train/seq.in','./dataset/train/seq.out')
    valid_feature, valid_label = read_file('./dataset/valid/seq.in','./dataset/valid/seq.out')
    # test_feature, test_label = read_file('./dataset/test/seq.in','./dataset/test/seq.out')

    vocab_words = get_vocab(train_feature)
    wvmodel = get_wvmodel()
    weight = get_weight(wvmodel)

    # 缺省值 [UNK] 填充值 [PAD]
    word2id = dict(zip(wvmodel.index2word,range(len(wvmodel.index2word))))                              # word -> id
    id2word = {idx:word for idx,word in enumerate(wvmodel.index2word)}     # id -> word
    word2id['[UNK]'] = len(word2id)    
    word2id['[PAD]'] = len(word2id)                          
    unk = word2id['[UNK]']              #UNK:低频词
    padding_value = word2id['[PAD]']    #PAD:填充词
    #获得标签字典
    label2id = {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8, 'START':9, 'STOP':10}
    # 词表生成 end

    # 统一句子长度
    # train_textlines = [line[:args.max_len] for line in train_feature]
    # valid_textlines = [line[:args.max_len] for line in valid_feature]

    # train_textlines = [line + ['[PAD]' for i in range(args.max_len-len(line))] for line in train_textlines]
    # valid_textlines = [line + ['[PAD]' for i in range(args.max_len-len(line))] for line in valid_textlines]
    # 转化为词表里面的index
    train_textlines = [[word2id[word] if word in word2id else unk for word in line] for line in train_feature]
    valid_textlines = [[word2id[word] if word in word2id else unk for word in line] for line in valid_feature]
    
    train_label = [[label2id[word] for word in line] for line in train_label]  
    valid_label = [[label2id[word] for word in line] for line in valid_label] 

    train_data = SentenceDataSet(train_textlines, train_label)
    valid_data = SentenceDataSet(valid_textlines, valid_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
        batch_size=args.batch_size,collate_fn=collate_fn, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=False)

    embedding_matrix = wvmodel.wv.vectors
    input_size = embedding_matrix.shape[0]   
    embedding_size = embedding_matrix.shape[1]  
    # pdb.set_trace()
    model = BiLSTM_CRF(input_size, len(word2id),label2id, embedding_matrix, embedding_size, args.hidden_size)

    if os.path.exists(args.param_path):
        print('loading params')
        # pdb.set_trace()
        model.load_state_dict(torch.load(args.param_path))

    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train(args, train_loader,model, optim, criterion)

    # embedding = nn.Embedding.from_pretrained(weight)
    # # requires_grad指定是否在训练过程中对词向量的权重进行微调
    # self.embedding.weight.requires_grad = True





