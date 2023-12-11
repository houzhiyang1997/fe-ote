import re
import math
import torch
import torch.nn as nn
from torchcrf import CRF


class Config:
    def __init__(self, args, num_outputs=3):
        self.embed_size = 300
        self.hidden_size = 150

        self.max_seq_length = args.max_length
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.num_layers = 1
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_outputs = num_outputs
        self.dropout = 0.5
        if args.model == 'LSTM':
            self.save_path = '../models/model_parameters/LSTM_parameter.bin'
        elif args.model == 'BiLSTM':
            self.save_path = '../models/model_parameters/BiLSTM_parameter.bin'
        elif args.model == 'GRU':
            self.save_path = '../models/model_parameters/GRU_parameter.bin'
        elif args.model == 'BiGRU':
            self.save_path = '../models/model_parameters/BiGRU_parameter.bin'
        elif args.model == 'LSTM_ATT':
            self.save_path = '../models/model_parameters/LSTM_ATT_parameter.bin'
        elif args.model == 'BiLSTM_ATT':
            self.save_path = '../models/model_parameters/BiLSTM_ATT_parameter.bin'
        elif args.model == 'GRU_ATT':
            self.save_path = '../models/model_parameters/GRU_ATT_parameter.bin'
        elif args.model == 'BiGRU_ATT':
            self.save_path = '../models/model_parameters/BiGRU_ATT_parameter.bin'

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


class BaselineModel(nn.Module):
    def __init__(self, args, embed, config):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
        self.dropout = nn.Dropout(config.dropout)

        if re.match('LSTM', args.model):
            self.baseline_model = nn.LSTM(config.embed_size, config.hidden_size,
                                          config.num_layers, bidirectional=False,
                                          batch_first=True, dropout=config.dropout)
            self.emission_layer = nn.Linear(config.hidden_size, config.num_outputs)
        elif re.match('BiLSTM', args.model):
            self.baseline_model = nn.LSTM(config.embed_size, config.hidden_size,
                                          config.num_layers, bidirectional=True,
                                          batch_first=True, dropout=config.dropout)
            self.emission_layer = nn.Linear(config.hidden_size * 2, config.num_outputs)
        elif re.match('GRU', args.model):
            self.baseline_model = nn.GRU(config.embed_size, config.hidden_size,
                                         config.num_layers, bidirectional=False,
                                         batch_first=True, dropout=config.dropout)
            self.emission_layer = nn.Linear(config.hidden_size, config.num_outputs)
        elif re.match('BiGRU', args.model):
            self.baseline_model = nn.GRU(config.embed_size, config.hidden_size,
                                         config.num_layers, bidirectional=True,
                                         batch_first=True, dropout=config.dropout)
            self.emission_layer = nn.Linear(config.hidden_size * 2, config.num_outputs)

        if re.search('ATT', args.model):
            if re.match('Bi', args.model):
                self.query = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)
                self.key = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)
                self.value = nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=False)
                self.sqrt_d = math.sqrt(config.hidden_size * 2)
            else:
                self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                self.sqrt_d = math.sqrt(config.hidden_size)

            self.softmax = nn.Softmax(dim=2)

        self.crf = CRF(config.num_outputs, batch_first=True)

    def forward(self, input_ids, input_masks, input_labels, mode='train'):
        embed = self.embedding(input_ids)

        # shape: (batch_size, max_length, hidden_size * 2 * bidirectional)
        out, _ = self.baseline_model(embed)
        out = self.dropout(out)

        if re.search('ATT', self.args.model):
            # shape: (batch_size, max_length, hidden_size * 2 * bidirectional)
            Q = self.query(out)
            K = self.key(out)
            V = self.value(out)
            # shape: (batch_size, max_length, max_length)
            alpha = self.softmax(
                torch.matmul(Q, K.permute(0, 2, 1)) / self.sqrt_d
            )
            # shape: (batch_size, max_length, hidden_size * 2 * bidirectional)
            att = torch.matmul(alpha, V)
            out = self.dropout(att)

        # shape: (batch_size, comment_max_length, num_outputs)
        emissions = self.emission_layer(out)

        if mode == 'train':
            loss = -self.crf(emissions, input_labels, mask=input_masks.byte())
            return loss

        loss = -self.crf(emissions, input_labels, mask=input_masks.byte())
        logits = self.crf.decode(emissions)
        return loss, logits
