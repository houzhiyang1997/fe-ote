import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertConfig, BertTokenizer, BertModel


class Config:
    def __init__(self, args, num_outputs):
        self.max_seq_length = args.max_length
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_outputs = num_outputs
        self.num_layers = 2
        self.dropout_rate = 0.5
        self.embed_size = 768
        self.hidden_size = 384
        self.model_path = '../static_data/bert-base-chinese'
        self.save_path = '../models/model_parameters/bert-crf_parameter.bin'

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


class Transformer(nn.Module):
    def __init__(self, args, config, device):
        super().__init__()
        self.args = args
        self.device = device
        self.config = config
        self.model_class, tokenizer_class, pretrained_weight = (BertModel,
                                                                BertTokenizer,
                                                                self.config.model_path)
        self.bert_config = BertConfig.from_pretrained(pretrained_weight)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        self.bert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)

        self.lstm = nn.LSTM(config.embed_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout_rate)
        self.lstm_fc = nn.Linear(config.embed_size * 2, config.num_outputs, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.emission_layer = nn.Linear(config.embed_size, config.num_outputs)
        self.crf = CRF(config.num_outputs, batch_first=True)

    def forward(self, input_comments, input_labels, mode='train'):
        tokens = self.tokenizer.batch_encode_plus(input_comments, add_special_tokens=True,
                                                  max_length=self.config.max_seq_length,
                                                  padding='max_length', truncation=True)
        ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)

        # shape: (batch_size, comment_max_length, embed_size)
        comment_hidden_state = self.bert(ids, attention_mask=attention_mask).last_hidden_state
        if self.args.is_seq == 'seq':
            # shape: (batch_size, comment_max_length, embed_size * 2)
            comment_lstm_out, _ = self.lstm(comment_hidden_state)
            comment_lstm_out = self.dropout(comment_lstm_out)
            # shape: (batch_size, comment_max_length, num_outputs)
            # comment_hidden_state = self.lstm_fc(comment_lstm_out)
            # shape: (batch_size, comment_max_length, num_outputs)
            # emissions = comment_hidden_state
            emissions = self.emission_layer(comment_lstm_out)
        else:
            comment_hidden_state = self.dropout(comment_hidden_state)
            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(comment_hidden_state)

        if mode == 'train':
            loss = -self.crf(emissions, input_labels, mask=attention_mask.byte())

            return loss

        loss = -self.crf(emissions, input_labels, mask=attention_mask.byte())
        logits = self.crf.decode(emissions)
        return loss, logits
