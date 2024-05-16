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
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from typing import List, Dict, Optional, Union
from torch.optim import AdamW, Adam, SGD, SparseAdam
from torch.utils.data import DataLoader
from edmsci.loss import Node2VecTripletLoss


class Word2Vec(nn.Module):
    def __init__(
        self, 
        vocab_size: Optional[int] = None, 
        embedding_size: Optional[int] = 100, 
        padding_idx: Optional[int] = None, 
        device: Optional[str] = "cpu", 
        learn_outvec=True):
        """
        Initialize the Word2Vec model.

        Args:
            vocab_size (int): The number of nodes or vocabularly.
            embedding_size (int): The dimensions of the embeddings.
            padding_idx (int): Index of the padding token.
            learn_outvec (bool): Whether to learn the output vectors. Defaults to True.
        """
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learn_outvec = learn_outvec
        self.multi_gpus_bool = False
        self.device = device
        self.devices = None
        self.to(device)


        if padding_idx is None:
            padding_idx = vocab_size

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
        


    def multi_gpus(self, devices: List[str] = None):
        self.multi_gpus_bool = True
        
        if devices is None:
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    devices = ["cuda:{}".format(i) for i in range(2)]
                else:
                    raise Exception("Not enough GPUs available.")
            else:
                raise Exception("No GPU available.")
        else:
            self.devices =  ["cuda:{}".format(i) for i in devices]

        print("input in-vectors to cuda:{}".format(devices[0]))
        self.ivectors.to(self.devices[0]) # put in-vectors into cuda:0
        print("input out-vectors to cuda:{}".format(devices[1]))
        self.ovectors.to(self.devices[1]) # put out-vectors into cuda:1

        
            

    def forward(self, data, forward_in = True):
        """
        Forward pass of the Word2vec model.

        Args:
            data: Input data.
            forward_in: Whether to return in-vectors. 
        """

        if self.multi_gpus_bool:
            if forward_in ==True:
                return self.ivectors(data.to(self.devices[0]))
            else:
                return self.ovectors(data.to(self.devices[1]))

        else:
            if forward_in == True:
                return self.ivectors(data)
            else:
                return self.ovectors(data)

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


    def fit(
        self,
        dataset,
        outputfile= None,
        checkpoint = 10000,
        batch_size: int = 256,
        learning_rate = 1e-3, 
        num_workers = 5):

        dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        loss_func = Node2VecTripletLoss(n_neg=1, multigpus= self.multi_gpus_bool, devices = self.devices)


        focal_params = filter(lambda p: p.requires_grad, self.parameters())
        # optim = SGD(focal_params, lr=learning_rate)
        # optim = Adam(focal_params, lr=learning_rate)
        optim = SparseAdam(focal_params, lr=learning_rate)

        pbar = tqdm(dataloader, miniters=10, total=len(dataloader))
        it = 0
        for params in pbar:
            # clear out the gradient
            optim.zero_grad()

            # compute the loss
            loss = loss_func(self, *params)

            # backpropagate
            loss.backward()

            # update the parameters
            optim.step()

            with torch.no_grad():
                pbar.set_postfix(loss=loss.item())

                if (it + 1) % checkpoint == 0:
                    if outputfile is not None:
                        torch.save(self.state_dict(), outputfile)
            it += 1
            torch.cuda.empty_cache()
        if outputfile is not None:
            torch.save(self.state_dict(), outputfile)
        self.eval()
        return self







    


    
