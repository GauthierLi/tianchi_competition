"""
    @author: gauthierLi
    @date: 04/18/2022
"""
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np

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
        self.inchannel = inchannel
        self.outchannel = outchannel
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
    def __init__(self, config_json, checkpoint, picture_path,
                 device=torch.device('cuda' if torch.cuda.is_available() else "cpu")):
        super(load_model, self).__init__()
        args = argparse.ArgumentParser(description="for load trained model")
        args.add_argument('-c', '--config', default=config_json, type=str,
                          help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=checkpoint, type=str,
                          help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                          help='indices of GPUs to enable (default: all)')
        self.args = args

        # transform the encoder
        self.device = device
        self.picture_path = picture_path
        self.config = ConfigParser.from_args(self.args)
        self.model = self.config.init_obj('arch', model_arch)
        self.state_dict = torch.load(checkpoint)["state_dict"]
        self._model_transform()

        # initial the conv parameter with trained kernels
        self.kernel_tensor = self._generate_kernel()
        self.conv = load_kernel_network(self.kernel_tensor)
        self.bn = nn.BatchNorm2d(self.conv.outchannel)
        self.relu = nn.LeakyReLU()

    def _model_transform(self):
        self.model.encoder = model_arch.kernel_generator(inchannel=3, stride=1, pad=1)
        self.model.load_state_dict(self.state_dict)
        self.model.decoder = nn.Sequential()
        self.model.to(self.device)

    def _generate_kernel(self):
        img = cv2.imread(self.picture_path).transpose((2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(dim=0).to(self.device)

        kernel = self.model(img)
        return kernel

    def forward(self, x):
        out = self.model(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    model_path = r"/media/gauthierli-org/GauLi/code/tainchi_competition/best_mdoel/models/0420_145455/config.json"
    checkpoint = r"/media/gauthierli-org/GauLi/code/tainchi_competition/best_mdoel/models/0420_145455/model_best.pth"
    picture_path = r"/media/gauthierli-org/GauLi/code/tainchi_competition/test_area/logo_imgs/1_RGB.jpg"
    img = cv2.imread(picture_path).transpose((2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(dim=0).to(device)
    print(img.size())
    tst_model = load_model(model_path, checkpoint, picture_path).to(device)
    print(tst_model(img).size())
