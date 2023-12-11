import re
import math
import torch
import torch.nn as nn
from torchcrf import CRF


class Config:
    def __init__(self, args, num_outputs=3):
        self.embed_size = 600
        self.lstm_hidden_size = 256

        self.max_seq_length = args.max_length
        self.lr = 1e-3
        self.weight_decay = 1e-4

        self.kernel_size = (3, 5, 7)

        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_outputs = num_outputs
        self.dropout = 0.5
        self.save_path = '../models/model_parameters/CombineLSTM_parameter.bin'

        self.tag2id = {
            'O': 0,
            'B': 1,
            'I': 2,
        }
        self.id2tag = {
            0: 'O',
            1: 'B',
            2: 'I',
        }


class CombineLSTM(nn.Module):
    def __init__(self, args, word2vec_embed, glove_embed, config):
        super().__init__()
        self.args = args
        self.word2vec_embedding = nn.Embedding.from_pretrained(word2vec_embed, freeze=False)
        self.glove_embedding = nn.Embedding.from_pretrained(glove_embed, freeze=False)
        # padding: 在文本长度方向上拼 padding, 不在词向量上拼 padding

        self.dense = nn.Linear(config.embed_size, config.embed_size)
        self.lstm = nn.LSTM(config.embed_size, config.lstm_hidden_size,
                            bidirectional=True, batch_first=True)

        self.emission_layer = nn.Linear(config.lstm_hidden_size * 2, config.num_outputs)
        self.crf = CRF(config.num_outputs, batch_first=True)

    def forward(self, word2vec_ids, glove_ids, input_masks, input_labels, mode='train'):
        word2vec_embed = self.word2vec_embedding(word2vec_ids)
        glove_embed = self.glove_embedding(glove_ids)
        # shape: (batch_size, max_length, embed_size)
        embed = torch.cat((word2vec_embed, glove_embed), 2)

        # shape: (batch_size, max_length, embed_size)
        linear_out = self.dense(embed)

        # shape: (batch_size, max_length, lstm_hidden_size * 2)
        lstm_hidden_state, _ = self.lstm(linear_out)

        # shape: (batch_size, comment_max_length, num_outputs)
        emissions = self.emission_layer(lstm_hidden_state)

        if mode == 'train':
            loss = -self.crf(emissions, input_labels, mask=input_masks.byte())
            return loss

        loss = -self.crf(emissions, input_labels, mask=input_masks.byte())
        logits = self.crf.decode(emissions)
        return loss, logits
