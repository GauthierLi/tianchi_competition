import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def crossEntropy(output, target):
    return F.cross_entropy(output, target)







