import torch
import numpy as np


class centerNetGT:
    # downsample rate
    R = 3.

    def __init__(self, img_size, bboxes):
        self.W, self.H = img_size
        # [[x, y, w/2, h/2, label], ...,]
        self.bboxes = bboxes
        # [50 -- > cls, 51 --> W, 52 --> H, 53 --> offset W, 54 --> offset H]
        self.gt = torch.zeros((54, int(self.W / self.R), int(self.H / self.R)), dtype=torch.float32)

        self.r = 4

    def generate_GT(self):
        for box in self.bboxes:
            x, y, w, h, label = box
            label = label - 1
            x_, y_, w_, h_ = int(x / self.R), int(y / self.R), int(w / self.R), int(h / self.R)

            self._caculate_r(w_, h_)
            self.center_gt(label, x_, y_)
            self.WH_gt(x_, y_, w_, h_)
            self.offset_gt(x, y, x_, y_)

    def center_gt(self, label, x, y):
        self.gt[label][x][y] = 1

    def WH_gt(self, x, y, w, h):
        for i in range(2 * self.r):
            for j in range(2 * self.r):
                self.gt[51][x - self.r + i][y - self.r + j] = w
                self.gt[52][x - self.r + i][y - self.r + j] = h

    def offset_gt(self, x, y, x_, y_):
        self.gt[53][x_][y_] = float(x) / float(self.R) - x_
        self.gt[54][x_][y_] = float(y) / float(self.R) - y_

    def _caculate_r(self, w, h, IoU=0.7):
        delta = 4 * (w + h) ** 2 - 16 * w * h * (1 - IoU) / (1 + IoU)
        self.r = int(w + h - np.sqrt(delta) / 2)
        if self.r % 2 != 0:
            self.r = self.r - 1
        self.r = np.max((0, self.r))
        return self.r

    def _gaussian_kernel(self):
        kernel = torch.zeros((1, self.r, self.r))

    def __call__(self):
        return self._gt

    @property
    def _gt(self):
        return self.gt

    @staticmethod
    def parse_to_standard(org_size, result):
        """
        format the predicted result to real picture
        @org_size: the origin size of picture_path
        @result: result after formate
        """
        pass


if __name__ == "__main__":
    tst_center_GT = centerNetGT((243, 243), bboxes=[[88, 88, 23, 30, 2]])
    print(tst_center_GT._caculate_r(20, 60))
