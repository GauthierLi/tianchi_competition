import cv2
import copy
import torch
import numpy as np


class centerNetGT:
    # downsample rate
    R = 3.

    def __init__(self, img_size):
        self.H, self.W = img_size
        # [[x, y, w/2, h/2, label], ...,]
        self.bboxes = []
        # [50 -- > cls, 51 --> H, 52 --> W, 53 --> offset H, 54 --> offset W]
        self.gt = torch.zeros((54, int(self.W / self.R), int(self.H / self.R)), dtype=torch.float32)

        self.r = 5

    def generate_GT(self):
        for box in self.bboxes:
            y, x, h, w, label = box  # x 行 y 列
            label = int(label) - 1
            x_, y_, h_, w_ = int(x / self.R), int(y / self.R), int(h / self.R), int(w / self.R)

            self._caculate_r(w_, h_)
            # print(f"r: {self.r}")
            self.center_gt(label, x_, y_)
            self.WH_gt(x_, y_, w_, h_)
            self.offset_gt(x, y, x_, y_)

    def center_gt(self, label, x, y):
        gaussian_conv = self._gaussian_kernel()
        tmp_gt = torch.zeros((int(self.W / self.R), int(self.H / self.R)))
        tmp_gt[x][y] = 1
        tmp_gt = gaussian_conv(tmp_gt.unsqueeze(dim=0).unsqueeze(dim=0)).squeeze().squeeze().numpy()
        gt = copy.deepcopy(self.gt[label]).numpy()

        # print(f"gt size: {tmp_gt.shape}, {gt.shape}")
        tmp_gt = np.maximum(tmp_gt, gt)
        self.gt[label] = torch.from_numpy(tmp_gt)

    def WH_gt(self, x, y, w, h):
        for i in range(2 * self.r + 1):
            for j in range(2 * self.r + 1):
                x__ = max(0, x - self.r + i)
                x__ = min(80, x__)
                y__ = max(0, y - self.r + j)
                y__ = min(80, y__)
                self.gt[50][x__][y__] = h
                self.gt[51][x__][y__] = w

    def offset_gt(self, x, y, x_, y_):
        for i in range(2 * self.r + 1):
            for j in range(2 * self.r + 1):
                x__ = max(0, x_ - self.r + i)
                x__ = min(80, x__)
                y__ = max(0, y_ - self.r + j)
                y__ = min(80, y__)
                self.gt[52][x__][y__] = float(x) / float(self.R) - x_
                self.gt[53][x__][y__] = float(y) / float(self.R) - y_

    def _caculate_r(self, w, h, IoU=0.7):
        delta = 4 * (w + h) ** 2 - 16 * w * h * (1 - IoU) / (1 + IoU)
        self.r = int(w + h - np.sqrt(delta) / 2)
        if self.r % 2 == 0:
            self.r = self.r - 1
        self.r = np.max((2, self.r))
        return self.r

    def _gaussian_kernel(self, sigma=7):
        ks = 2 * self.r + 1
        kernel = torch.zeros((ks, ks))
        for i in range(ks):
            for j in range(ks):
                x = -self.r + i
                y = -self.r + j
                kernel[i][j] = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * sigma))
        kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
        gaussian_conv = torch.nn.Conv2d(1, 1, kernel_size=ks, stride=1, padding=int((ks - 1) / 2),
                                        bias=0).requires_grad_(False)
        gaussian_conv.weight.data = kernel
        return gaussian_conv

    def __call__(self, bboxes):
        self.bboxes = bboxes
        self.generate_GT()
        return self._gt

    @property
    def _gt(self):
        return self.gt

    @staticmethod
    def parse_to_standard(org_size, result, R=3, belief=0.9):
        """
        format the predicted result to real picture
        @org_size: the origin size of picture_path (h, w)
        @result: size [54, w/R, h/R]
        """
        bbox = []
        result = result.numpy()
        result[result < belief] = 0
        H, W = result[0].shape

        for label in range(50):
            for i in range(H):
                for j in range(W):
                    around = result[label][max(0, i - 2):min(i + 3, H - 1), max(0, j - 2):
                                                                                      min(j + 3, W - 1)]
                    around = around < result[label][i][j]
                    count = around.sum()
                    around_count = around.shape[0] * around.shape[1]
                    if count > around_count * 0.95:
                        x = j
                        y = i
                        x = (result[53][i][j] + x) * R
                        y = (result[52][i][j] + y) * R
                        h = (result[50][i][j]) * R
                        w = (result[51][i][j]) * R
                        box = [x, y, h, w, label + 1]
                        bbox.append(box)
        return bbox


if __name__ == "__main__":
    tst_center_GT = centerNetGT((243, 243))
    # kernel = tst_center_GT._gaussian_kernel(5)
    bbox = np.array([[210, 98, 32, 18, 1], [98, 98, 24, 19, 1]])
    gt = tst_center_GT(bbox)
    bbox1 = tst_center_GT.parse_to_standard(1, gt)
    print(bbox1)
    fram = gt.numpy()[0]
    cv2.imshow("img", fram)
    cv2.waitKey()
