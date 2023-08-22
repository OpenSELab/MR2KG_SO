import logging

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


# single BERT
class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()

        # BERT encoder
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        # 设置参数的可修改的
        for param in self.encoder.parameters():
            param.requires_grad = True

        # 全连接层，将特征向量维度为（hidden_size）的
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, labels=None):
        output = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        last_hidden_state, pooler_output = output[0], output[1]
        lg = self.fc(pooler_output)
        prob = torch.softmax(lg, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lg, labels)
            return loss, prob
        else:
            return prob

