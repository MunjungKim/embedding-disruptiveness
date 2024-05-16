# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by: Munjung Kim
# @Last Modified Time: 2024-05-16 15:27:29

import torch
import torch.nn as nn
import torch.nn.functional as F


class Node2VecTripletLoss(nn.Module):
    def __init__(self, n_neg, multigpus = False, devices = None):
        super(Node2VecTripletLoss, self).__init__()
        self.n_neg = n_neg
        self.multigpus = multigpus
        self.devices = devices
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, model, iwords, owords, nwords):

        if self.multigpus ==False:

            ivectors = model.forward_i(iwords).unsqueeze(2)
            ovectors = model.forward_o(owords)
            nvectors = model.forward_o(nwords).neg()
            oloss = self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(dim=1)
            nloss = (
                self.logsigmoid(torch.bmm(nvectors, ivectors).squeeze())
                .view((-1, owords.size()[1], self.n_neg))
                .sum(dim=2)
                .mean(dim=1)
            )
            return -(oloss + nloss).mean()

        else:
            ivectors = model.forward(iwords.to(self.devices[0])).unsqueeze(2).to(self.devices[1]) # put target words into cuda:0 and transfer its vectors in cuda:0 into cuda:1
            ovectors = model.forward(owords.to(self.devices[1]),forward_in = False)
            nvectors = model.forward(nwords.to(self.devices[1]),forward_in=False).neg()
            if self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).dim() == 1:
                oloss = self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(dim=0)
                
            else:
                oloss = self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(dim=1)
        
            nloss = (
                self.logsigmoid(torch.bmm(nvectors, ivectors).squeeze())
                .view(-1, owords.size()[1], self.n_neg)
                .sum(dim=2)
                .mean(dim=1)
            )
    
        
        return -(oloss + nloss).mean()



class ModularityTripletLoss(nn.Module):
    def __init__(self, n_neg):
        super(ModularityTripletLoss, self).__init__()
        self.n_neg = n_neg
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, model, iwords, owords, nwords, base_iwords, base_owords):
        ivectors = model.forward_i(iwords).unsqueeze(2)
        ovectors = model.forward_o(owords)
        nvectors = model.forward_o(nwords).neg()

        base_ivectors = model.forward_i(base_iwords).unsqueeze(2)
        base_ovectors = model.forward_o(base_owords)

        oloss = torch.bmm(ovectors, ivectors).squeeze().mean(dim=1)
        nloss = (
            torch.bmm(nvectors, ivectors)
            .squeeze()
            .view(-1, owords.size()[1], self.n_neg)
            .sum(dim=2)
            .mean(dim=1)
        )

        base_loss = torch.bmm(base_ovectors, base_ivectors).squeeze().mean(dim=1)

        loss = -(oloss + nloss - 0.5 * torch.pow(base_loss, 2)).mean()

        return loss


