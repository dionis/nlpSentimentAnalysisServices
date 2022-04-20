# -*- coding: utf-8 -*-
# file: aen.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from ..layers.dynamic_rnn import DynamicLSTM
from ..layers.squeeze_embedding import SqueezeEmbedding
from ..layers.attention import Attention, NoQueryAttention
from ..layers.point_wise_feed_forward import PositionwiseFeedForward
import torch
import torch.nn as nn
import torch.nn.functional as F


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class AEN_HAT_GloVe(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AEN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()

        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out



class AEN_BERT_HAT(nn.Module): #Attentional Encoder Network for Targeted Sentiment Classiﬁcation
    def __init__(self, bert, opt):
        super(AEN_BERT_HAT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.gate = torch.nn.Sigmoid()

        self.ec0 = torch.nn.Embedding(self.opt.taskcla, opt.hidden_dim)
        self.ec1 = torch.nn.Embedding(self.opt.taskcla, opt.hidden_dim)
        self.ec2 = torch.nn.Embedding(self.opt.taskcla, opt.hidden_dim)
        self.ec3 = torch.nn.Embedding(self.opt.taskcla, opt.hidden_dim)
        self.ec4 = torch.nn.Embedding(self.opt.taskcla, opt.hidden_dim)

        self.hat = True

        self.last = torch.nn.ModuleList()
        for t in range(self.opt.taskcla):
            self.last.append(nn.Linear(opt.hidden_dim*3, opt.polarities_dim))

    def forward(self,t, inputs, s):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        # context, _ = self.bert(context, output_all_encoded_layers=False)
        context, _,  _ = self.bert(context)
        context = self.dropout(context)
        target = self.squeeze_embedding(target, target_len)

        masks = self.mask(t, s=s)
        gc0, gc1, gc2, gc3, gc4 = masks

        # , output_all_encoded_layers = False
        target, _,  _ = self.bert(target)
        target = self.dropout(target)

        # Zero hat apply
        hc, _ = self.attn_k(context, context)
        hc = hc * gc0.view(1, 1, -1).expand_as(hc)

        # First hat apply
        hc = self.ffn_c(hc)
        hc = hc* gc1.view(1, 1, -1).expand_as(hc)

        # Second hat apply
        ht, _ = self.attn_q(context, target)
        ht = ht * gc2.view(1, 1, -1).expand_as(ht)

        # Third hat apply
        ht = self.ffn_t(ht)
        ht = ht * gc3.view(1, 1, -1).expand_as(ht)

        # Four hat apply
        s1, _ = self.attn_s1(hc, ht)
        s1 = s1 * gc4.view(1, 1, -1).expand_as(s1)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        #torch.tensor(context_len, dtype=torch.float).to(self.opt.device)

        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)
        #torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)

        y = []
        for i, _ in enumerate(range(self.opt.taskcla)):
            y.append(self.last[i]( x))

        return y, masks


    def mask(self, t, s=1):
        gc0 = self.gate(s * self.ec0(t))
        gc1 = self.gate(s * self.ec1(t))
        gc2 = self.gate(s * self.ec2(t))
        gc3 = self.gate(s * self.ec3(t))
        gc4 = self.gate(s * self.ec4(t))

        return [gc0, gc1, gc2, gc3, gc4]

    def get_view_for(self, n, masks):
        # Variable == > attn_k.weight
        # Variable == > attn_k.w_k.weight
        # Variable == > attn_k.w_k.bias
        # Variable == > attn_k.w_q.weight
        # Variable == > attn_k.w_q.bias
        # Variable == > attn_k.proj.weight
        # Variable == > attn_k.proj.bias
        # Variable == > attn_q.weight
        # Variable == > attn_q.w_k.weight
        # Variable == > attn_q.w_k.bias
        # Variable == > attn_q.w_q.weight
        # Variable == > attn_q.w_q.bias
        # Variable == > attn_q.proj.weight
        # Variable == > attn_q.proj.bias
        # Variable == > ffn_c.w_1.weight
        # Variable == > ffn_c.w_1.bias
        # Variable == > ffn_c.w_2.weight
        # Variable == > ffn_c.w_2.bias
        # Variable == > ffn_t.w_1.weight
        # Variable == > ffn_t.w_1.bias
        # Variable == > ffn_t.w_2.weight
        # Variable == > ffn_t.w_2.bias
        # Variable == > attn_s1.weight
        # Variable == > attn_s1.w_k.weight
        # Variable == > attn_s1.w_k.bias
        # Variable == > attn_s1.w_q.weight
        # Variable == > attn_s1.w_q.bias
        # Variable == > attn_s1.proj.weight
        # Variable == > attn_s1.proj.bias
        gc0, gc1, gc2, gc3, gc4 = masks

        # if n == "attn_k.weight":
        #     return gc0.data.view(-1, 1).expand_as(self.attn_k.weight)
        # if n == "attn_k.w_k.weight":
        #     return gc0.data.view(-1, 1).expand_as(self.attn_k.w_k.weight)
        # if n == "attn_k.w_q.weight":
        #     return gc0.data.view(-1, 1).expand_as(self.attn_k.w_q.weight)
        if n == "attn_k.proj.weight":
            return gc0.data.view(-1, 1).expand_as(self.attn_k.proj.weight)
        elif n == 'attn_k.w_k.bias':
            return gc0.data.view(-1)
        elif n == 'attn_k.w_q.bias':
            return gc0.data.view(-1)
        elif n == 'attn_k.proj.bias':
            return gc0.data.view(-1)


        # if n == "attn_q.weight":
        #     return gc1.data.view(-1, 1).expand_as(self.attn_q.weight)
        # if n == "attn_q.w_k.weight":
        #     return gc1.data.view(-1, 1).expand_as(self.attn_q.w_k.weight)
        # if n == "attn_q.w_q.weight":
        #     return gc1.data.view(-1, 1).expand_as(self.attn_q.w_q.weight)
        if n == "attn_q.proj.weight":
            return gc1.data.view(-1, 1).expand_as(self.attn_q.proj.weight)
        elif n == 'attn_q.w_k.bias':
            return gc1.data.view(-1)
        elif n == 'attn_q.w_q.bias':
            return gc1.data.view(-1)
        elif n == 'attn_q.proj.bias':
            return gc1.data.view(-1)



        elif n == 'ffn_c.w_1.weight':
            post = gc2.data.view(-1, 1).expand_as(self.ffn_c.w_1.weight)
            pre = gc2.data.view(1, -1,1).expand_as(self.ffn_c.w_1.weight)
            return torch.min(post, pre)
        elif n == 'ffn_c.w_2.weight':
            post = gc2.data.view(-1, 1).expand_as(self.ffn_c.w_2.weight)
            pre = gc2.data.view(1, -1,1).expand_as(self.ffn_c.w_2.weight)
            return torch.min(post, pre)
        elif n == 'ffn_c.w_1.bias':
            return gc2.data.view(-1)
        elif n == 'ffn_c.w_2.bias':
            return gc2.data.view(-1)

        elif n == 'ffn_t.w_1.weight':
            post = gc3.data.view(-1, 1).expand_as(self.ffn_t.w_1.weight)
            pre = gc3.data.view(1, -1, 1).expand_as(self.ffn_t.w_1.weight)
            return torch.min(post, pre)
        elif n == 'ffn_t.w_2.weight':
            post = gc3.data.view(-1, 1).expand_as(self.ffn_t.w_2.weight)
            pre = gc3.data.view(1, -1,1).expand_as(self.ffn_t.w_2.weight)
            return torch.min(post, pre)
        elif n == 'ffn_t.w_1.bias':
            return gc3.data.view(-1)
        elif n == 'ffn_t.w_2.bias':
            return gc3.data.view(-1)

        # elif n == 'attn_s1.weight':
        #     post = gc4.data.view(-1).expand_as(self.attn_s1.weight)
        #     pre = gc4.data.view(1, -1,1).expand_as(self.attn_s1.weight)
        #     return torch.min(post, pre)
        elif n == 'attn_s1.w_k.weight':
            post = gc4.data.view(-1).expand_as(self.attn_s1.w_k.weight)
            pre = gc4.data.view(1, -1).expand_as(self.attn_s1.w_k.weight)
            return torch.min(post, pre)
        elif n == 'attn_s1.w_q.weight':
            post = gc4.data.view(-1).expand_as(self.attn_s1.w_q.weight)
            pre = gc4.data.view(1, -1).expand_as(self.attn_s1.w_q.weight)
        # elif n == 'attn_s1.proj.weight':
        #     post = gc4.data.view(-1).expand_as(self.attn_s1.proj.weight)
        #     pre = gc4.data.view(1, -1).expand_as(self.attn_s1.proj.weight)
        #     return torch.min(post, pre)
        elif n == 'attn_s1.w_k.bias':
            return gc4.data.view(-1)
        elif n == 'attn_s1.w_q.bias':
            return gc4.data.view(-1)
        elif n == 'attn_s1.proj.bias':
            return gc4.data.view(-1)

        return None

    def get_bert_model_parameters(self):
        variable_name = ["attn_k", "attn_q", "ffn_c", "ffn_t","attn_s1","last"]
        modelVariables = []

        for i, (name, var) in enumerate(self.named_parameters()):
            for iname in variable_name:
                if name.find(iname) != -1:
                    modelVariables.append((name, var))
                    break

        return modelVariables