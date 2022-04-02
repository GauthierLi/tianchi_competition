import os
import cv2
import torch
import numpy as np

def draw_bbox(img, bboxAndlabel):
    """
        @bboxAndlabel: [[[centerx, centery, h, w], label], ... ...]
    """
    for boxandlabel in bboxAndlabel:
        print(type(boxandlabel[0][0]))
        box = [i.numpy()[0] if isinstance(i, torch.Tensor) else i for i in boxandlabel[0]]
        text_info = str(boxandlabel[1].numpy()[0]) if isinstance(boxandlabel[1][0], torch.Tensor) else str(boxandlabel[1])
        text_len = len(text_info)
        x1, y1, x2, y2 = [int(box[0] - box[2]), int(box[1] - box[3]), int(box[0] + box[2]), int(box[1] + box[3])]
        img = cv2.rectangle(img, (x1, y1),
                            (x2, y2), (255, 0, 0), 2)
        img = cv2.rectangle(img, (int(box[0] - box[2]), int(box[1] - box[3]) - 16),
                            (int(box[0] - box[2]) + text_len * 11, int(box[1] - box[3])), (250, 250, 20), -1)
        img = cv2.putText(img, text_info, (int(box[0] - box[2]), int(box[1] - box[3]) - 5), cv2.FONT_HERSHEY_COMPLEX,
                          0.5, (0, 0, 0))
    cv2.imshow("img", img)
    cv2.waitKey(0)