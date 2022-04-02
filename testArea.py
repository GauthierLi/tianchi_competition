"""
    @author:gauthierLi
    @data:03/27/2022
    @func:test only
"""
import os
import cv2
import torch
import numpy as np
import pycocotools.coco as coco
from data_loader.coco import coco_dataloader


def _get_format_bbox(label_info):
    bbox_lst = []
    for item in label_info:
        category_id = item['category_id']
        box = item["bbox"]
        format_box = [(box[0] + box[2] / 2, box[1] + box[3] / 2, box[2] / 2, box[3] / 2), category_id]
        bbox_lst.append(format_box)
    return bbox_lst


def draw_bbox(img, bboxAndlabel):
    for boxandlabel in bboxAndlabel:
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


if __name__ == "__main__":
    dataLoader = coco_dataloader(f"/mnt/data/coco/", 1, training=None)
    for img, bboxAndlabel in dataLoader:
        print(bboxAndlabel[0].data.item)  # [[[tensor([73.5100], dtype=torch.float64), tensor([240.], dtype=torch.float64), tensor([70.2700], dtype=torch.float64), tensor([38.3800], dtype=torch.float64)], tensor([23])]]
        img = img.squeeze().numpy()
        draw_bbox(img, bboxAndlabel)