import logging

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class BTModel(nn.Module):
    def __init__(self, textEncoder, codeEncoder,
                 text_hidden_size, code_hidden_size, num_class):
        super(BTModel, self).__init__()
        self.textEncoder = textEncoder
        self.codeEncoder = codeEncoder
        self.text_hidden_size = text_hidden_size
        self.code_hidden_size = code_hidden_size
        self.num_class = num_class
        for param in self.textEncoder.parameters():
            param.requires_grad = True
        for param in self.codeEncoder.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(text_hidden_size + code_hidden_size, num_class)

    def forward(self, text_input_ids=None, code_input_ids=None, labels=None):
        text_output = self.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]  # [batch_size, hiddensize]
        code_output = self.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))[1]
        # print(text_output, text_output.shape)  # [batchsize, hiddensize]
        # print(code_output, code_output.shape)
        # 将text_output 与 code_output 合并，batch_size不动，只在hiddensize上合并
        combine_output = torch.cat([text_output, code_output], dim=-1)
        # print('combine_output.shape:',combine_output.shape)
        logits = self.fc(combine_output)
        prob = torch.softmax(logits, -1)
        # print('logits:',logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            # print(lg)
            # print(labels)
            # print('logits:', logits)
            # print('labels:', labels)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
