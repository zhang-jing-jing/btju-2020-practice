import torch
import torch.nn as nn
from layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, replace_masked, get_mask

#  参考https://github.com/coetaur0/ESIM/blob/master/esim/model.py
class ESIM(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings_matrix=None,
                 padding_value=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
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

        self._word_embedding = nn.Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        padding_idx=padding_value,
                                        _weight=embeddings_matrix)

        self._word_embedding.weight.requires_grad = False

        if self.dropout:
            self._rnn_dropout = RNNDropout(p = self.dropout)
        
        self._encoding = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                          self.hidden_size,
                                          self.hidden_size,
                                          bidirectional= True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))
        # 初始化 所有的权重和偏置
        self.apply(_init_esim_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        """
        :param premises: 前提 A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
        :param premises_lengths:A 1D tensor containing the lengths of the
                premises in 'premises'.
        :param hypotheses:假设  A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
        :param hypothese_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.
        :return:
            logits：tensor (batch, num_classes) 包含每个输出类的logits
            probabilities: tensor (batch, num_classes) 包含每个类的概率
        """
        premises_mask = get_mask(premises,premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses,hypotheses_lengths).to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,hypotheses_lengths)

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises,
                            premises_mask,
                            encoded_hypotheses,
                            hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises*attended_premises],
                                      dim=-1)# dim=-1 按列拼接
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                       attended_hypotheses,
                                       encoded_hypotheses - attended_hypotheses,
                                       encoded_hypotheses * attended_hypotheses],
                                      dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(encoded_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises,premises_lengths)
        v_bj = self._composition(projected_hypotheses,hypotheses_lengths)

        v_a_avg = torch.sum(v_ai*premises_mask.unsqueeze(1).transpose(2,1),dim=1)\
                  /torch.sum(premises_mask, dim=1,keepdim=True) # 在第二维增加维度，第三个维度和第二个维度交换

        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1).transpose(2, 1), dim=1) \
                  / torch.sum(hypotheses_mask, dim=1, keepdim=True)  # 在第二维增加维度，第三个维度和第二个维度交换

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return  logits, probabilities

    def _init_esim_weights(module) :
        """
        初始化ESIM模型的权重和偏置
        """
        if isinstance(module, nn.Linear):
            # xavier_uniform_ 保证输入输出方差一致
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)

        elif isinstance(module, nn.LSTM):
            nn.init.xavier_normal_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0.0)
            nn.init.constant_(module.bias_hh_l0.data, 0.0)
            hidden_size = module.bias_hh_l0.data.shape[0]  // 4
            module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

            if(module.bidirectional):
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0.reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0


