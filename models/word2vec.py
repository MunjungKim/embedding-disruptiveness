# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Munjung Kim
# @Last Modified time: 2023-06-03 23:47:35

"""
The original code is in https://github.com/skojaku/node2vec/blob/master/node2vecs/torch/models.py
"""


import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, learn_outvec=True):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learn_outvec = learn_outvec
        self.ivectors = nn.Embedding(
            self.vocab_size + 1,
            self.embedding_size,
            padding_idx=padding_idx,
            sparse=True,
        )
        self.ovectors = nn.Embedding(
            self.vocab_size + 1,
            self.embedding_size,
            padding_idx=padding_idx,
            sparse=True,
        )
        torch.nn.init.uniform_(
            self.ovectors.weight, -0.5 / embedding_size, 0.5 / embedding_size
        )
        torch.nn.init.uniform_(
            self.ivectors.weight, -0.5 / embedding_size, 0.5 / embedding_size
        )
        print("input in vectors to cuda 0")
        self.ivectors.to('cuda:0') # put invector into the first machine
        print("input in vectors to cuda 1")
        self.ovectors.to('cuda:1')# put outvector into the second machine

    def forward(self, data, forward_in = True):
        """
        Forward pass of the Word2vec model.

        Args:
            data: Input data.
            forward_in: Whether to return in-vectors. 
        """

        if forward_in ==True:
            return self.ivectors(data.to('cuda:0'))
        else:
            return self.ovectors(data.to('cuda:1'))

    def embedding(self, return_out_vector=False):
        if return_out_vector is False:
            if self.ivectors.weight.is_cuda:
                return self.ivectors.weight.data.cpu().numpy()[: self.vocab_size]
            else:
                return self.ivectors.weight.data.numpy()[: self.vocab_size]
        else:
            if self.ovectors.weight.is_cuda:
                return self.ovectors.weight.data.cpu().numpy()[: self.vocab_size]
            else:
                return self.ovectors.weight.data.numpy()[: self.vocab_size]
