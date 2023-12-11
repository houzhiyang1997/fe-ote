import os
import torch
import torch.nn as nn
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
        # self.model_path = r'H:\huggingface\bert-base-chinese'
        self.model_path = '../static_data/bert-base-chinese'
        self.save_path = '../models/model_parameters/TBXC_{}_{}_parameter.bin'.format(args.ablation,
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


class Ablation(nn.Module):
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

        self.cbert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)
        self.dropout = nn.Dropout(config.dropout_rate)

        if args.interaction_mode == 'mlp_concat':
            self.emission_layer = nn.Linear(config.embed_size * 2, config.num_outputs)
        else:
            self.emission_layer = nn.Linear(config.embed_size, config.num_outputs)
        if args.ablation == 'wce':
            self.wbert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)
            for name, parameter in self.wbert.named_parameters():
                if name != 'pooler.dense.weight' and name != 'pooler.dense.bias':
                    parameter.requires_grad_ = False

            self.dense = nn.Linear(config.embed_size * 2, config.embed_size)

        if args.interaction_mode == 'mlp':
            self.gelu = nn.GELU()
            self.weibo_linear = nn.Linear(config.embed_size,
                                          config.embed_size,
                                          bias=False)
            self.comment_linear = nn.Linear(config.embed_size,
                                            config.embed_size,
                                            bias=False)
            self.bias = nn.Parameter(torch.zeros(config.embed_size, ))

        if args.interaction_mode == 'concat_mlp':
            self.dense = nn.Linear(config.embed_size * 2, config.embed_size)

        if args.interaction_mode == 'mlp_concat':
            self.relu = nn.ReLU()
            self.weibo_linear = nn.Linear(config.embed_size, config.embed_size)
            self.comment_linear = nn.Linear(config.embed_size, config.embed_size)

    def interaction(self, weibo_embedding, comment_embedding):
        if self.args.interaction_mode == 'dot' or self.args.interaction_mode == 'minus_dot':

            # shape: (batch_size, comment_max_length, embed_size)
            if self.args.interaction_mode == 'dot':
                interaction_hidden_state = torch.mul(weibo_embedding, comment_embedding)
            else:
                interaction_hidden_state = -torch.mul(weibo_embedding, comment_embedding)

            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

        elif self.args.interaction_mode == 'minus':

            # shape: (batch_size, comment_max_length, embed_size)
            interaction_hidden_state = torch.sub(comment_embedding, weibo_embedding)
            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

        elif self.args.interaction_mode == 'mlp':

            # shape: (batch_size, comment_max_length, embed_size)
            interaction_hidden_state = self.gelu(self.weibo_linear(weibo_embedding) +
                                                 self.comment_linear(comment_embedding) +
                                                 self.bias)

            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

        elif self.args.interaction_mode == 'concat_mlp':

            concat_embedding = torch.cat((comment_embedding, weibo_embedding), 2)

            interaction_hidden_state = self.dense(concat_embedding)

            interaction_hidden_state = self.dropout(interaction_hidden_state)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

        elif self.args.interaction_mode == 'mlp_concat':

            weibo_embedding = self.weibo_linear(weibo_embedding)
            comment_embedding = self.comment_linear(comment_embedding)
            interaction_hidden_state = torch.cat((weibo_embedding, comment_embedding), 2)

            # shape: (batch_size, comment_max_length, num_outputs)
            emissions = self.emission_layer(interaction_hidden_state)

        return emissions

    def forward(self, input_weibos, input_comments):
        """
        :param input_weibos: list
                length: batch_size
        :param input_comments: list
                length: batch_size
        """
        comment_tokens = self.tokenizer.batch_encode_plus(input_comments, add_special_tokens=True,
                                                          max_length=self.config.comment_max_seq_length,
                                                          padding='max_length', truncation=True)
        comment_ids = torch.tensor(comment_tokens['input_ids']).to(self.device)
        comment_attention_mask = torch.tensor(comment_tokens['attention_mask']).to(self.device)

        # shape: (batch_size, comment_max_length, embed_size)
        comment_hidden_state = self.cbert(comment_ids, attention_mask=comment_attention_mask).last_hidden_state

        if self.args.ablation == 'wce':
            weibo_tokens = self.tokenizer.batch_encode_plus(input_weibos, add_special_tokens=True,
                                                            max_length=self.config.weibo_max_seq_length,
                                                            padding='max_length', truncation=True)
            weibo_ids = torch.tensor(weibo_tokens['input_ids']).to(self.device)
            weibo_attention_mask = torch.tensor(weibo_tokens['attention_mask']).to(self.device)

            # shape: (batch_size, embed_size)
            weibo_sentence_embedding = self.wbert(weibo_ids, attention_mask=weibo_attention_mask).pooler_output
            # shape: (batch_size, 1, embed_size)
            weibo_sentence_embedding = weibo_sentence_embedding.unsqueeze(1)
            weibo_sentence_embedding = weibo_sentence_embedding.repeat(1, comment_hidden_state.shape[1], 1)

            emissions = self.interaction(weibo_sentence_embedding, comment_hidden_state)

        else:
            emissions = self.emission_layer(comment_hidden_state)

        return emissions
