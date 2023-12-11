import os
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertConfig, BertTokenizer, BertModel


class Config:
    def __init__(self, args, num_outputs):
        self.comment_max_seq_length = args.comment_max_length
        self.weibo_max_seq_length = args.weibo_max_length

        self.bert_embed_size = 768
        self.word2vec_embed_size = 300
        self.lstm_hidden_size = 256
        if args.mode == 'cbert' or args.mode == 'bbert':
            self.lr = 1e-5
        elif args.mode == 'wbert' or args.mode == 'blstm':
            self.lr = 1e-3

        self.weight_decay = 1e-4
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_outputs = num_outputs
        self.dropout_rate = 0.5

        # self.model_path = r'H:\huggingface\bert-base-chinese'
        self.model_path = '../static_data/bert-base-chinese'
        self.save_path = '../models/model_parameters/interaction_{}_{}_parameter.bin'.format(args.mode,
                                                                                             args.interaction_mode)

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


class Interaction(nn.Module):
    def __init__(self, args, config, device, embed=None):
        super().__init__()
        self.args = args
        self.device = device
        self.config = config

        if args.mode == 'cbert':
            self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
            self.model_class, tokenizer_class, pretrained_weight = (BertModel,
                                                                    BertTokenizer,
                                                                    self.config.model_path)
            self.bert_config = BertConfig.from_pretrained(pretrained_weight)
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
            self.cbert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)
            self.wlstm = nn.LSTM(config.word2vec_embed_size, config.lstm_hidden_size,
                                 bidirectional=True, batch_first=True)
            self.c_dense = nn.Linear(config.bert_embed_size, config.lstm_hidden_size * 2)

        elif args.mode == 'bbert':
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
        elif args.mode == 'wbert':
            self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
            self.model_class, tokenizer_class, pretrained_weight = (BertModel,
                                                                    BertTokenizer,
                                                                    self.config.model_path)
            self.bert_config = BertConfig.from_pretrained(pretrained_weight)
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
            self.wbert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)
            self.clstm = nn.LSTM(config.word2vec_embed_size, config.lstm_hidden_size,
                                 bidirectional=True, batch_first=True)

            self.w_dense = nn.Linear(config.bert_embed_size, config.lstm_hidden_size * 2)

            for name, parameter in self.wbert.named_parameters():
                if name != 'pooler.dense.weight' and name != 'pooler.dense.bias':
                    parameter.requires_grad_ = False

        elif args.mode == 'blstm':
            self.embedding = nn.Embedding.from_pretrained(embed, freeze=False)
            self.wlstm = nn.LSTM(config.word2vec_embed_size, config.lstm_hidden_size,
                                 bidirectional=True, batch_first=True)
            self.clstm = nn.LSTM(config.word2vec_embed_size, config.lstm_hidden_size,
                                 bidirectional=True, batch_first=True)

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

        if args.interaction_mode == 'concat_mlp':
            self.dense = nn.Linear(config.lstm_hidden_size * 4, config.lstm_hidden_size * 2)

        if args.interaction_mode == 'mlp_concat':
            self.gelu = nn.GELU()
            self.relu = nn.ReLU()
            self.weibo_linear = nn.Linear(config.lstm_hidden_size * 2, config.lstm_hidden_size * 2)
            self.comment_linear = nn.Linear(config.lstm_hidden_size * 2, config.lstm_hidden_size * 2)
            self.bias = nn.Parameter(torch.zeros(config.lstm_hidden_size * 2, ))
        if args.mode == 'bbert':
            self.emission_layer = nn.Linear(config.bert_embed_size, config.num_outputs)
        elif args.interaction_mode == 'mlp_concat':
            self.emission_layer = nn.Linear(config.lstm_hidden_size * 4, config.num_outputs)
        else:
            self.emission_layer = nn.Linear(config.lstm_hidden_size * 2, config.num_outputs)

        # self.emission_layer = nn.Linear(config.embed_size, config.num_outputs)
        self.crf = CRF(config.num_outputs, batch_first=True)

    def forward_bbert(self, input_weibos, input_comments):
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

        # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
        # shape: (batch_size, comment_max_length, embed_size)
        weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_hidden_state.shape[1], 1)

        return weibo_sentence_embedding, comment_hidden_state, comment_attention_mask

    def forward_cbert(self, input_weibos, input_comments, input_masks):
        weibo_embed = self.embedding(input_weibos)
        weibo_lstm_out, _ = self.wlstm(weibo_embed)
        # shape: (batch_size, lstm_hidden_size * 2)
        weibo_sentence_embedding = weibo_lstm_out[:, -1, :]

        # shape: (batch_size, 1, embed_size)
        weibo_sentence_embedding = weibo_sentence_embedding.unsqueeze(1)

        comment_ids = torch.tensor(input_comments).to(self.device)
        comment_attention_mask = torch.tensor(input_masks).to(self.device)

        # shape: (batch_size, comment_max_length, embed_size)
        comment_hidden_state = self.cbert(comment_ids, attention_mask=comment_attention_mask).last_hidden_state

        # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
        # shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
        weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_hidden_state.shape[1], 1)

        return weibo_sentence_embedding, comment_hidden_state, comment_attention_mask

    def forward_wbert(self, input_weibos, input_comments, input_weibo_masks, input_comment_masks):
        comment_embed = self.embedding(input_comments)
        # shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
        comment_lstm_out, _ = self.clstm(comment_embed)

        weibo_ids = torch.tensor(input_weibos).to(self.device)
        weibo_attention_mask = torch.tensor(input_weibo_masks).to(self.device)

        # shape: (batch_size, bert_embed_size)
        weibo_sentence_embedding = self.wbert(weibo_ids,
                                              attention_mask=weibo_attention_mask).pooler_output

        # shape: (batch_size, 1, bert_embed_size)
        weibo_sentence_embedding = weibo_sentence_embedding.unsqueeze(1)

        # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
        # shape: (batch_size, comment_max_length, bert_embed_size)
        weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_lstm_out.shape[1], 1)

        return weibo_sentence_embedding, comment_lstm_out, input_comment_masks

    def forward_blstm(self, input_weibos, input_comments, input_masks):
        weibo_embed = self.embedding(input_weibos)

        weibo_lstm_out, _ = self.wlstm(weibo_embed)
        # shape: (batch_size, lstm_hidden_size * 2)
        weibo_sentence_embedding = weibo_lstm_out[:, -1, :]

        # shape: (batch_size, 1, embed_size)
        weibo_sentence_embedding = weibo_sentence_embedding.unsqueeze(1)

        comment_embed = self.embedding(input_comments)
        # shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
        comment_lstm_out, _ = self.clstm(comment_embed)

        # 将每个 batch 的句向量给重复 comment_max_length, 用于后面对位相乘
        # shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
        weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_lstm_out.shape[1], 1)

        return weibo_sentence_embedding, comment_lstm_out, input_masks

    def interaction_layer(self, weibo_sentence_embedding, comment_hidden_state,
                          input_labels, comment_attention_mask, mode):
        if self.args.interaction_mode == 'dot' or self.args.interaction_mode == 'minus_dot':

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

        elif self.args.interaction_mode == 'concat_mlp':

            concat_embedding = torch.cat((comment_hidden_state, weibo_sentence_embedding), 2)

            interaction_hidden_state = self.dense(concat_embedding)

            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

            if mode == 'train':
                loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())

                return loss

            loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())
            logits = self.crf.decode(emissions)
            return loss, logits

        elif self.args.interaction_mode == 'mlp_concat':

            weibo_embedding = self.weibo_linear(weibo_sentence_embedding)
            comment_embedding = self.comment_linear(comment_hidden_state)
            interaction_hidden_state = torch.cat((weibo_embedding, comment_embedding), 2)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

            if mode == 'train':
                loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())

                return loss

            loss = -self.crf(emissions, input_labels, mask=comment_attention_mask.byte())
            logits = self.crf.decode(emissions)
            return loss, logits

    def forward(self, input_weibos, input_comments, input_labels,
                mode='train', input_masks=None, input_weibo_masks=None):
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
        if self.args.mode == 'bbert':
            # w_sentence_embedding shape: (batch_size, comment_max_length, bert_embed_size)
            # c_sentence_embedding shape: (batch_size, comment_max_length, bert_embed_size)
            w_sentence_embedding, c_hidden_state, c_attention_mask = self.forward_bbert(input_weibos,
                                                                                        input_comments)
        elif self.args.mode == 'cbert':
            input_weibos = input_weibos.to(self.device)
            input_comments = input_comments.to(self.device)
            input_masks = input_masks.to(self.device)

            # w_sentence_embedding shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
            # c_sentence_embedding shape: (batch_size, comment_max_length, bert_embed_size)
            w_sentence_embedding, c_hidden_state, c_attention_mask = self.forward_cbert(input_weibos,
                                                                                        input_comments,
                                                                                        input_masks)

            # shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
            c_hidden_state = self.c_dense(c_hidden_state)

        elif self.args.mode == 'wbert':
            input_weibos = input_weibos.to(self.device)
            input_comments = input_comments.to(self.device)
            input_masks = input_masks.to(self.device)
            input_weibo_masks = input_weibo_masks.to(self.device)

            # w_sentence_embedding shape: (batch_size, comment_max_length, bert_embed_size)
            # c_sentence_embedding shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
            w_sentence_embedding, c_hidden_state, c_attention_mask = self.forward_wbert(input_weibos,
                                                                                        input_comments,
                                                                                        input_weibo_masks,
                                                                                        input_masks)

            # shape: (batch_size, comment_max_length, lstm_hidden_size * 2)
            w_sentence_embedding = self.w_dense(w_sentence_embedding)

        elif self.args.mode == 'blstm':
            input_weibos = input_weibos.to(self.device)
            input_comments = input_comments.to(self.device)
            input_masks = input_masks.to(self.device)
            w_sentence_embedding, c_hidden_state, c_attention_mask = self.forward_blstm(input_weibos,
                                                                                        input_comments,
                                                                                        input_masks)

        if mode == 'train':
            loss = self.interaction_layer(w_sentence_embedding,
                                          c_hidden_state,
                                          input_labels,
                                          c_attention_mask,
                                          mode)

            return loss

        loss, logits = self.interaction_layer(w_sentence_embedding,
                                              c_hidden_state,
                                              input_labels,
                                              c_attention_mask,
                                              mode)
        return loss, logits
