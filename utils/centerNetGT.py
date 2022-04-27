import torch
import numpy as numpy

class centerNetGT:
    #downsample rate
    R = 3
    def __init__(self, img_size, bboxes):
        self.W, self.H = img_size
        # [[x, y, w/2, h/2, label], ...,]
        self.bboxes = bboxes
        # [50 -- > cls, 51 --> W, 52 --> H, 53 --> offset W, 54 --> offset H]
        self.gt = torch.zeros((54, int(self.W/self.R), int(self.H/self.R)), dtype=torch.float32)

    def generate_GT(self):
        for box in self.bboxes:
            x, y, w, h, label = box
            label = label - 1
            x_, y_, w_, h_ = int(x/self.R), int(y/self.R), int(w/self.R), int(h/self.R)
            self.center_gt(label, x_, y_)
            self.WH_gt(x_, y_, w_, h_)
            self.offset_gt(x, y, x_, y_)

    def center_gt(self, label, x, y):
        self.gt[label][x][y] = 1


    def WH_gt(self, x, y, w, h):
        pass

    def offset_gt(self, x, y, x_, y_):
        pass

    def _caculate_r(self):
        pass

    @property
    def _gt(self):
        return self.gt

    def __call__(self):
        return self._gt