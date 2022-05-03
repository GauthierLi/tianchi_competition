"""
    @author: gauthierLi
    @date: 04/02/2022
"""
import os
import cv2
import json
import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base import BaseModel
from parse_config import ConfigParser
from model.model_resnet import resnet50


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- basic blocks ---------------------------------------------------------
class convBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, pad, kernel_size=3):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        oup = self.conv(x)
        oup = self.bn(oup)
        oup = self.relu(oup)
        oup = self.dropout(oup)
        return oup


class downSample(nn.Module):
    def __init__(self, pading=1, kernel_size=3, stride=3):
        super(downSample, self).__init__()
        self.downsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pading)

    def forward(self, x):
        return self.downsample(x)


class oneSizeConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(oneSizeConv, self).__init__()
        self.CBR = convBlock(inchannel, outchannel, stride=1, pad=0, kernel_size=1)

    def forward(self, x):
        return self.CBR(x)


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden, oup_dim):
        super(MLP, self).__init__()
        self.inp = nn.Linear(inp_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu1 = nn.LeakyReLU()
        self.hidden = nn.Linear(hidden, oup_dim)
        self.bn2 = nn.BatchNorm1d(oup_dim)
        self.relu2 = nn.LeakyReLU()
        # self.sftmax = nn.Softmax(1)

    def forward(self, x):
        oup = self.inp(x)
        oup = self.bn1(oup)
        oup = self.relu1(oup)
        oup = self.hidden(oup)
        oup = self.bn2(oup)
        oup = self.relu2(oup)
        # oup = self.sftmax(oup)

        return oup


class Bottleneck(nn.Module):
    expansion = 3

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ---------------------------------------------- end basic blocks ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- pretrain kernel extract network --------------------------------------------


class chain_process(nn.Module):
    def __init__(self, inchannel, stride, pad, edge_size, layer_depth):
        super(chain_process, self).__init__()
        self.generate_kernel = True if stride == 3 and pad == 0 else False
        self.inchannel = inchannel
        self.stride = stride
        self.pad = pad
        self.edge_size = edge_size
        self.depth = layer_depth
        self.seq = self.generate_from_chains()

    def stright_chain(self, lent):
        seq_lst = [nn.Sequential(Bottleneck(self.inchannel * (3 ** i), self.inchannel * (3 ** i)),
                                 convBlock(inchannel=self.inchannel * (3 ** (i + 1)),
                                           outchannel=self.inchannel * (3 ** (i + 1)),
                                           stride=self.stride, pad=self.pad))
                   for i in
                   range(lent)]
        return nn.Sequential(*seq_lst)

    def generate_from_chains(self):
        seq_lst = [downSample() if self.generate_kernel else nn.Sequential() for j in range(self.depth)]
        seq = nn.Sequential(*seq_lst, self.stright_chain(self.edge_size - self.depth))
        # print(f"num of chains:{len(chains)}, chains", chains)
        return seq

    def forward(self, x):
        return self.seq(x)


class kernel_generator(nn.Module):

    def __init__(self, inchannel, stride, pad, edge_size=4):
        super(kernel_generator, self).__init__()
        self.stride = stride
        self.pad = pad

        self.inchannel = inchannel
        self.basic_block = nn.Sequential()
        self.edge_size = edge_size

        self.chain0 = chain_process(self.inchannel, self.stride, self.pad, self.edge_size, 0)
        self.chain1 = chain_process(self.inchannel, self.stride, self.pad, self.edge_size, 1)
        self.chain2 = chain_process(self.inchannel, self.stride, self.pad, self.edge_size, 2)
        self.chain3 = chain_process(self.inchannel, self.stride, self.pad, self.edge_size, 3)
        self.chain4 = chain_process(self.inchannel, self.stride, self.pad, self.edge_size, 4)

        self.chains = [self.chain0, self.chain1, self.chain2, self.chain3, self.chain4]

        self.oneSizeConv1 = oneSizeConv(inchannel=363, outchannel=127)

        self.oneSizeConv2 = oneSizeConv(inchannel=127, outchannel=9)
        self.oneSizeConv3 = oneSizeConv(inchannel=9, outchannel=127)

    def forward(self, x):
        heads = []
        for chain in self.chains:
            head = chain(x)
            heads.append(head)
        oup = torch.cat(heads, 1)
        oup = self.oneSizeConv1(oup)

        oup = self.oneSizeConv2(oup)
        oup = self.oneSizeConv3(oup)

        return oup


class classify_decoder(nn.Module):
    def __init__(self, num_cls, inchannel=127):
        super(classify_decoder, self).__init__()
        self.num_cls = num_cls
        self.CBR = convBlock(inchannel=inchannel, outchannel=inchannel, stride=1, pad=0)
        self.MLP = MLP(inchannel, 64, num_cls)

    def forward(self, x):
        oup = self.CBR(x)
        oup = oup.view(-1, 127)
        oup = self.MLP(oup)
        return oup


class kernel_extract_network(nn.Module):
    r"""
    use to pretrain the kernel_extract net
    """

    def __init__(self, inchannel=3, num_cls=50):
        super(kernel_extract_network, self).__init__()
        self.encoder = kernel_generator(inchannel=inchannel, stride=3, pad=0)
        self.decoder = classify_decoder(num_cls)

    def forward(self, x):
        oup = self.encoder(x)
        oup = self.decoder(oup)
        return oup


class load_kernel_network(nn.Module):
    r"""
    initial the convNet with the specific kernel tensor
    """

    def __init__(self, kernel_tensor, stride=1, padding=1):
        super(load_kernel_network, self).__init__()
        self.kernel_tensor = kernel_tensor
        B, C, W, H = self.kernel_tensor.size()
        if not self.kernel_tensor.requires_grad:
            self.kernel_tensor.requires_grad = True
        self.conv = nn.Conv2d(in_channels=C, out_channels=B, kernel_size=(W, H), stride=stride, padding=padding)
        self.conv.weight.data = self.kernel_tensor
        self.bn = nn.BatchNorm2d(B)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# ---------------------------------------------------  end  ------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
"""
1， 127 张 feature map 太多了，需要去冗余
2， 设计 detection 网络和 ground truth 
"""


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- detection model ----------------------------------------------------------
class initial_preprocess_networks(nn.Module):
    """
    generate b*50*H*W feature map
    """

    def __init__(self, config_path, resume, average_img_dir):
        super(initial_preprocess_networks, self).__init__()
        conf_file = open(config_path, 'r')
        self.conf = json.load(conf_file)
        self.model = eval(self.conf["arch"]["type"])()
        self.eval_model = eval(self.conf["arch"]["type"])()
        self.eval_model.encoder = kernel_generator(inchannel=3, stride=1, pad=1)

        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        self.model.load_state_dict(state_dict)
        self.eval_model.load_state_dict(state_dict)

        self.model.decoder = nn.Sequential()
        self.model.encoder.oneSizeConv3 = nn.Sequential()
        self.eval_model.decoder = nn.Sequential()
        self.eval_model.encoder.oneSizeConv3 = nn.Sequential()

        self.img_dir = average_img_dir
        self.generated_kernel = self._generate_kernel_from_imgs()
        self.conv = load_kernel_network(self.generated_kernel)
        self.bn = nn.BatchNorm2d(50)
        self.relu = nn.LeakyReLU()

    def _generate_kernel_from_imgs(self):
        device = self.model.encoder.oneSizeConv2.CBR.conv.weight.device
        _, _, img_lst = list(os.walk(self.img_dir))[0]
        kernel_list = []
        for img_path in img_lst:
            abs_img_path = os.path.join(self.img_dir, img_path)
            label = eval(img_path.split('_')[0])
            img = cv2.imread(abs_img_path).astype(np.float32).transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(dim=0).to(device)
            tmp_kernel = self.model(img)
            kernel_list.append(tmp_kernel)
        kernels = torch.cat(kernel_list, 0)
        return kernels

    def forward(self, x):
        out = self.eval_model(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


def res50():
    return resnet50(include_top=False)

class detector(nn.Module):
    def __init__(self, config:str, resume:str, reference_path:str):
        super(detector, self).__init__()
        self.init_kernel_network = initial_preprocess_networks(config, resume, reference_path)
        self.res50 = res50()

    def forward(self, x):
        out = self.init_kernel_network(x)
        out = self.res50(out)
        return out

if __name__ == "__main__":
    device = torch.device('cuda')
    config = r"../saved/models/kernel_generator/0422_150723/config.json"
    resume = r"../saved/models/kernel_generator/0422_150723/model_best.pth"
    reference_path = r"../test_area/logo_imgs"
    tst = detector(config, resume, reference_path).to(device)

    img = r"/media/gauthierli-org/GauLi/code/tainchi_competition/test_area/logo_imgs/3_RGB.jpg"
    img = cv2.imread(img).astype(np.float32).transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(dim=0).to(device)
    print(img.shape)

    img = tst(img)

    print(img.size())
