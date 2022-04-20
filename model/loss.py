import torch.nn.functional as F
import torch
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def crossEntropy(output, target):
    return F.cross_entropy(output, target)


def focal_loss(output, target):
    criterion = FocalLoss()
    return criterion(output, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=4, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, input, target):
        # self.alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        BCELoss = F.cross_entropy(input, target, reduce=False)
        pt = torch.exp(-BCELoss)
        FC_loss = self.alpha * torch.pow((1 - pt), self.gamma) * BCELoss

        if self.reduce:
            return torch.mean(FC_loss)
        else:
            return FC_loss
