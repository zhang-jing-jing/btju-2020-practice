import torch
import torch.nn as nn
from layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention


class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, embeddings_matrix=None, padding_value=0, dropout=0.5, num_classes=3, device="cpu"):
        """
        vocab_size: 词典size
        embedding_dim: 词嵌入维度
        hidden_size: 隐层
        embeddings_matrix: size (vocab_size, embedding_dim)
        padding_value: 填充值
        dropout: 辍学率
        num_classes：最后输出的分类数
        device：计算的设备 
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self.word_embeds = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.word_embeds.from_pretrained(embeddings_matrix)
        self.word_embeds.weight.requires_grad = False

        if self.dropout:
            self._rnn_dropout = RNNDropout(p = self.dropout)
        
        self.encoding = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
        # self.encoding = Seq2SeqEncoder()
        self.attention = SoftmaxAttention()
        
        
