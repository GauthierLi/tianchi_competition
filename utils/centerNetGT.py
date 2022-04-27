import torch
import numpy as numpy

class centerNetGT:
    def __init__(self, img_size, bboxes):
        self.W, self.H = img_size
        # [[x, y, w/2, h/2, label], ...,]
        self.bboxes = bboxes
        self.gt = torch.zeros((54, self.W, self.H), dtype=torch.float32)

    def center_gt(self):
        pass

    def WH_gt(self):
        pass

    def offset_gt(self):
        pass

    @property
    def _gt(self):
        return self.gt

    def __call__(self):
        return self._gt