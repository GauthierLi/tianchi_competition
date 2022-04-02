"""
    @author: gauthierLi
    @date: 04/02/2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class convBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, pad):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x):
        oup = self.conv(x)
        oup = self.bn(oup)
        oup = self.relu(oup)
        return oup


class downSample(nn.Module):
    def __init__(self, pading=1, kernel_size=3, stride=3):
        super(downSample, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pading)

    def forward(self, x):
        return self.downsample(x)


class kernel_generator(nn.Module):
    def __init__(self, inchannel, edge_size=5):
        super(kernel_generator, self).__init__()
        self.inchannel = inchannel
        self.basic_block = nn.Sequential()
        self.edge_size = edge_size

    def generate_dla_chains(self, lent):
        chains = []
        for i in range(lent + 1):
            tmp_seq_lst = [downSample() for j in range(i)]
            tmp_seq = nn.Sequential(*tmp_seq_lst, self.stright_chain(lent - i))
            chains.append(tmp_seq)
        # print(f"num of chains:{len(chains)}, chains", chains)
        return chains

    def stright_chain(self, lent):
        seq_lst = [
            convBlock(inchannel=self.inchannel * (3 ** i), outchannel=self.inchannel * (3 ** (i + 1)), stride=3, pad=0)
            for i in
            range(lent)]
        return nn.Sequential(*seq_lst)

    def forward(self, x):
        heads = []
        chains = self.generate_dla_chains(self.edge_size)
        for chain in chains:
            head = chain(x)
            heads.append(head)

        oup = torch.cat(heads, 1)

        return oup
