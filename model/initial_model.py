"""
    @author: gauthierLi
    @date: 04/18/2022
"""
import torch
import argparse
import torch.nn as nn

import torch.nn.functional as F
import model as model_arch

from utils import read_json
from parse_config import ConfigParser


class load_kernel_network(nn.Module):
    r"""
    initial the convNet with the specific kernel tensor
    """

    def __init__(self, kernel_tensor, inchannel=127, outchannel=127):
        super(load_kernel_network, self).__init__()
        self.kernel_tensor = kernel_tensor
        if not self.kernel_tensor.requires_grad:
            self.kernel_tensor.requires_grad = True
        self.conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class load_model(nn.Module):
    def __init__(self, config_json, checkpoint, picture_path, device=torch.device('cuda')):
        super(load_model, self).__init__()
        args = argparse.ArgumentParser(description="for load trained model")
        args.add_argument('-c', '--config', default=config_json, type=str,
                          help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=checkpoint, type=str,
                          help='path to latest checkpoint (default: None)')
        self.args = args.parse_args()

        # transform the encoder
        self.device = device
        self.config = ConfigParser.from_args(self.args)
        self.model = self.config.init_obj('arch', model_arch)
        self.state_dict = torch.load(checkpoint)["state_dict"]
        self._model_transform()

        # initial the conv parameter with trained kernels
        self.kernel_tensor = self._generate_kernel()
        self.kernel_generate = load_kernel_network(self.kernel_tensor)

    def _model_transform(self):
        self.model.encoder = model_arch.kernel_generator(inchannel=3, stride=1, pad=1)
        self.model.load_state_dict(self.state_dict)
        self.mdoel.decoder = nn.Sequential()
        self.mdoel.to(self.device)

    def _generate_kernel(self):
        pass
