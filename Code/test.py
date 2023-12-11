import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer


class Config:
    def __init__(self, args, num_outputs, max_seq_length=64, batch_size=64):

        self.max_seq_length = max_seq_length
        self.weight_decay = 1e-4
        self.batch_size = batch_size
        self.num_outputs = num_outputs


class BERT(nn.Module):
    def __init__(self, args, config, device):
        super().__init__()
        self.args = args
        self.device = device
        self.common_config = config
        self.model_class, tokenizer_class, pretrained_weight = (AutoModelForSequenceClassification,
                                                                AutoTokenizer,
                                                                self.common_config.model_path)
        self.bert_config = AutoConfig.from_pretrained(pretrained_weight,
                                                      num_labels=self.common_config.num_outputs)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        self.bert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)

    def forward(self, inputs):
        tokens = self.tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                                  max_length=self.common_config.max_seq_length,
                                                  padding=True, truncation=True)
        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        return outputs.logits
