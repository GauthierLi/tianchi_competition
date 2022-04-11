import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet34
from torch import tensor


class extract_kernel_network():
    r"""
        get 7380 kernels from the origin gray scale photo sized with (1, 243, 243).

        **Example**:
            >>> k1 = extract_kernel_network()
            >>> t1 = torch.rand((1, 243, 243))
            >>> kernels = k1(t1)
    """

    def __init__(self):
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=3)

    def _get_down_lst(self, img):
        img = F.normalize(img)
        img_lst = [img]
        size = img.size()[-1]
        iteras_time = int(np.log(size) / np.log(3))
        for i in range(iteras_time - 1):
            img_lst.append(self.downsample(img_lst[-1]))
        return img_lst

    def slice_kernel(self, img):
        kernel = []
        size = img.size()[-1]
        log_size = int(size / 3)
        for i in range(log_size):
            for j in range(log_size):
                kernel.append(img[:, 3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)])
        return kernel

    def __call__(self, img):
        down_imgs = self._get_down_lst(img)
        kernel_slice = []
        for down_img in down_imgs:
            kernel_slice += self.slice_kernel(down_img)
        kernel_slice = torch.stack(kernel_slice)
        kernel_slice.requires_grad = True
        return kernel_slice


class init_conv_network(nn.Module):
    def __init__(self, gray_imgs_path, in_channel=1):
        super(init_conv_network, self).__init__()
        self.conv = nn.Conv2d(in_channel, 7380, kernel_size=3, stride=1, padding=1)
        self.gray_imgs_path = gray_imgs_path
        self.kernel_extract = extract_kernel_network()
        self.bn = nn.BatchNorm2d(7380)
        self.relu = nn.LeakyReLU()


        self.kernel_initialator()

    def kernel_initialator(self):
        img = cv2.imread(self.gray_imgs_path, cv2.IMREAD_GRAYSCALE).astype('float32')
        img = torch.from_numpy(img).unsqueeze(dim=0)
        kernel = self.kernel_extract(img)
        self.conv.weight.data = kernel

    def forward(self, img):
        img = F.normalize(img)
        out = self.conv(img)
        out = self.bn(out)
        out = self.relu(out)
        return out


if __name__ == '__main__':
    """
    problem is kernels are too much, how to decrease them?
    - ignore them
    - 1 * 1 size convolutional
    """
    device = torch.device("cuda")
    t1 = init_conv_network(r"logo_imgs/gray_label/Honda 1_GRAY.jpg").to(device)
    # path to source image Samples
    tst_pic = r"/home/gauthierli-org/data/data/vehicle-logos-dataset/Source Image Samples/logo thumb5.png"
    tst_pic = cv2.imread(tst_pic, cv2.IMREAD_GRAYSCALE).astype("float32")
    tst_pic = torch.from_numpy(tst_pic).to(device)
    tst_pic = tst_pic.unsqueeze(dim=0)
    tst_pic = F.normalize(tst_pic)

    fram = t1(tst_pic).cpu().data.numpy()[0]
    print(fram)
    cv2.imshow("fram", fram)
    cv2.waitKey()



