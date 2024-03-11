import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1,position=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList(
        [nn.Linear(d_model, self.num_heads * self.head_size) for _ in range(3)]
    )
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.position = position
        if(position):
            self.pe = PositionalEncoding(d_model=d_model,dropout=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value :(bsz,d_model,h,w)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        bsz,ch,h,w = query.shape
        query = query.permute(0,2,3,1).contiguous().view(bsz, -1, ch) # bsz, h*w, d_model
        key = key.permute(0,2,3,1).contiguous().view(bsz, -1, ch)
        value = value.permute(0,2,3,1).contiguous().view(bsz, -1, ch)

        if(self.position):
            query = self.pe(query)
            key = self.pe(key)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_, key_, value_ = [
            layer(x)
            .view(bsz, h*w, self.num_heads, self.head_size)
            .transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ] # bsz,num_head,hw,d_k

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.view(bsz, ch, h, w)
        # 3) "Concat" using a view and apply a final linear.
        #return torch.mean(x, -3)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])