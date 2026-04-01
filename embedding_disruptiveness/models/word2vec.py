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


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, device_in='cuda:0', device_out='cuda:1', learn_outvec=True, dtype_reduce=False):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learn_outvec = learn_outvec
        self.dtype_reduce = dtype_reduce
        self.device_in = torch.device(device_in)
        self.device_out = torch.device(device_out)

        if not dtype_reduce:
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
        else:
            self.ivectors = nn.Embedding(
                self.vocab_size + 1,
                self.embedding_size,
                padding_idx=padding_idx,
                sparse=True,
            ).to(dtype=torch.bfloat16)
            self.ovectors = nn.Embedding(
                self.vocab_size + 1,
                self.embedding_size,
                padding_idx=padding_idx,
                sparse=True,
            ).to(dtype=torch.bfloat16)

        torch.nn.init.uniform_(
            self.ovectors.weight, -0.5 / embedding_size, 0.5 / embedding_size
        )
        torch.nn.init.uniform_(
            self.ivectors.weight, -0.5 / embedding_size, 0.5 / embedding_size
        )
        self.ivectors.to(self.device_in)
        self.ovectors.to(self.device_out)

    def forward(self, data, forward_in=True):
        """
        Forward pass of the Word2vec model.

        Args:
            data: Input data.
            forward_in: Whether to return in-vectors.
        """
        if forward_in:
            return self.ivectors(data.to(self.device_in))
        else:
            return self.ovectors(data.to(self.device_out))
        
        

    def embedding(self, return_out_vector=False):
        if not self.dtype_reduce:
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
        else:
            if return_out_vector is False:
                if self.ivectors.weight.is_cuda:
                    return self.ivectors.weight.data.cpu().to(torch.bfloat16).numpy()[: self.vocab_size]
                else:
                    return self.ivectors.weight.data.to(torch.bfloat16).numpy()[: self.vocab_size]
            else:
                if self.ovectors.weight.is_cuda:
                    return self.ovectors.weight.data.cpu().to(torch.bfloat16).numpy()[: self.vocab_size]
                else:
                    return self.ovectors.weight.data.to(torch.bfloat16).numpy()[: self.vocab_size]