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
        self.dropout = 0.3
        self.save_path = '../models/model_parameters/CCG_parameter.bin'

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


class CCG(nn.Module):
    def __init__(self, args, embed, config):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
        self.dropout = nn.Dropout(config.dropout)

        self.baseline_model = nn.LSTM(config.embed_size, config.hidden_size,
                                      bidirectional=True, batch_first=True)
        self.emission_layer = nn.Linear(config.hidden_size * 2, config.num_outputs)

        self.crf = CRF(config.num_outputs, batch_first=True)

    def forward(self, input_ids, input_masks, input_labels, mode='train'):

        embed = self.embedding(input_ids)
        embed = self.dropout(embed)

        # shape: (batch_size, max_length, hidden_size * 2 * bidirectional)
        out, _ = self.baseline_model(embed)
        out = self.dropout(out)

        # shape: (batch_size, comment_max_length, num_outputs)
        emissions = self.emission_layer(out)

        if mode == 'train':
            loss = -self.crf(emissions, input_labels, mask=input_masks.byte())
            return loss

        loss = -self.crf(emissions, input_labels, mask=input_masks.byte())
        logits = self.crf.decode(emissions)
        return loss, logits
