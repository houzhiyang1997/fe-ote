import os
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertConfig, BertTokenizer, BertModel


class Config:
    def __init__(self, args, num_outputs):
        self.comment_max_seq_length = args.comment_max_length
        self.weibo_max_seq_length = args.weibo_max_length
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_outputs = num_outputs
        self.dropout_rate = 0.5
        self.embed_size = 768
        self.model_path = '../static_data/bert-base-chinese'
        self.save_path = '../models/model_parameters/my_TBXC_{}_parameter.bin'.format(args.interaction_mode)

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


class TBXC(nn.Module):
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
        self.wbert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)
        self.cbert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)
        for name, parameter in self.wbert.named_parameters():
            if name != 'pooler.dense.weight' and name != 'pooler.dense.bias':
                parameter.requires_grad_ = False

        self.dropout = nn.Dropout(config.dropout_rate)

        if args.interaction_mode == 'mlp':
            self.gelu = nn.GELU()
            self.weibo_linear = nn.Linear(config.lstm_hidden_size * 2,
                                          config.lstm_hidden_size * 2,
                                          bias=False)
            self.comment_linear = nn.Linear(config.lstm_hidden_size * 2,
                                            config.lstm_hidden_size * 2,
                                            bias=False)
            self.bias = nn.Parameter(torch.zeros(config.lstm_hidden_size * 2, ))

        elif args.interaction_mode == 'my_mlp':
            self.my_linear = nn.Linear(config.embed_size * 2, config.embed_size, bias=False)
        elif args.interaction_mode == 'my_mlp2':
            self.relu = nn.ReLU()
            self.my_weibo_linear = nn.Linear(config.embed_size, config.embed_size, bias=False)
            self.my_comment_linear = nn.Linear(config.embed_size, config.embed_size, bias=False)
            self.bias = nn.Parameter(torch.zeros(config.embed_size, ))

        if args.interaction_mode == 'my_mlp2':
            self.emission_layer = nn.Linear(config.embed_size * 2, config.num_outputs)
        else:
            self.emission_layer = nn.Linear(config.embed_size, config.num_outputs)
        self.crf = CRF(config.num_outputs, batch_first=True)

    def forward(self, input_weibos, input_comments, input_labels, mode='train'):
        """
        :param input_weibos: list
                length: batch_size
        :param input_comments: list
                length: batch_size
        :param input_labels: LongTensor
                length: (batch_size, seq_length)
        :param mode: str
                'train', 'dev', 'test'
                'train': only output log likelihood
                'dev' and 'test': output log likelihood and decoder output
        """
        weibo_tokens = self.tokenizer.batch_encode_plus(input_weibos, add_special_tokens=True,
                                                        max_length=self.config.weibo_max_seq_length,
                                                        padding='max_length', truncation=True)
        weibo_ids = torch.tensor(weibo_tokens['input_ids']).to(self.device)
        weibo_attention_mask = torch.tensor(weibo_tokens['attention_mask']).to(self.device)

        comment_tokens = self.tokenizer.batch_encode_plus(input_comments, add_special_tokens=True,
                                                          max_length=self.config.comment_max_seq_length,
                                                          padding='max_length', truncation=True)
        comment_ids = torch.tensor(comment_tokens['input_ids']).to(self.device)
        comment_attention_mask = torch.tensor(comment_tokens['attention_mask']).to(self.device)

        # shape: (batch_size, embed_size)
        weibo_sentence_embedding = self.wbert(weibo_ids, attention_mask=weibo_attention_mask).pooler_output
        # shape: (batch_size, comment_max_length, embed_size)
        comment_hidden_state = self.cbert(comment_ids, attention_mask=comment_attention_mask).last_hidden_state
        # shape: (batch_size, 1, embed_size)
        weibo_sentence_embedding = weibo_sentence_embedding.unsqueeze(1)

        if self.args.interaction_mode == 'dot' or self.args.interaction_mode == 'minus_dot':

            # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
            # shape: (batch_size, comment_max_length, embed_size)
            weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_hidden_state.shape[1], 1)

            # shape: (batch_size, comment_max_length, embed_size)
            if self.args.interaction_mode == 'dot':
                interaction_hidden_state = torch.mul(weibo_sentence_embedding, comment_hidden_state)
            else:
                interaction_hidden_state = -torch.mul(weibo_sentence_embedding, comment_hidden_state)

            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

            if mode == 'train':
                loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())

                return loss

            loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())
            logits = self.crf.decode(emissions)
            return loss, logits

        elif self.args.interaction_mode == 'minus':
            # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
            # shape: (batch_size, comment_max_length, embed_size)
            weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_hidden_state.shape[1], 1)

            # shape: (batch_size, comment_max_length, embed_size)
            interaction_hidden_state = torch.sub(comment_hidden_state, weibo_sentence_embedding)
            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

            if mode == 'train':
                loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())

                return loss

            loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())
            logits = self.crf.decode(emissions)
            return loss, logits

        elif self.args.interaction_mode == 'mlp':
            # shape: (batch_size, comment_max_length, embed_size)
            interaction_hidden_state = self.gelu(self.weibo_linear(weibo_sentence_embedding) +
                                                 self.comment_linear(comment_hidden_state) +
                                                 self.bias)

            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

            if mode == 'train':
                loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())

                return loss

            loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())
            logits = self.crf.decode(emissions)
            return loss, logits
        elif self.args.interaction_mode == 'my_mlp':
            # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
            # shape: (batch_size, comment_max_length, embed_size)
            weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_hidden_state.shape[1], 1)
            # shape: (batch_size, comment_max_length, embed_size * 2)
            interaction_hidden_state = torch.cat((comment_hidden_state, weibo_sentence_embedding), 2)
            # shape: (batch_size, comment_max_length, embed_size)
            interaction_hidden_state = self.my_linear(interaction_hidden_state)


            # shape: (batch_size, comment_max_length, embed_size)
            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

            if mode == 'train':
                loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())

                return loss

            loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())
            logits = self.crf.decode(emissions)
            return loss, logits
        elif self.args.interaction_mode == 'my_mlp2':
            # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
            # shape: (batch_size, comment_max_length, embed_size)
            weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_hidden_state.shape[1], 1)
            # shape: (batch_size, comment_max_length, embed_size)
            weibo_embedding = self.my_weibo_linear(weibo_sentence_embedding)
            comment_embedding = self.my_comment_linear(comment_hidden_state)
            interaction_hidden_state = torch.cat((weibo_embedding, comment_embedding), 2)
            # interaction_hidden_state = self.relu(self.my_weibo_linear(weibo_sentence_embedding) +
            #                                      self.my_comment_linear(comment_hidden_state) +
            #                                      self.bias)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

            if mode == 'train':
                loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())

                return loss

            loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())
            logits = self.crf.decode(emissions)
            return loss, logits