"""
    @author:gauthierLi
    @date:03/07/2022
"""
import os
import cv2
import numpy as np
import torchvision.transforms
from PIL import Image
import pycocotools.coco as coco
import torch
import torchvision.transforms as T

from base import BaseDataLoader
from torch.utils.data import Dataset
from utils.centerNetGT import centerNetGT
from torch.utils.data.dataloader import default_collate


def collate(batch):
    imgs, bboxes = [], []
    for img, bbox, gt in batch:
        imgs.append(np.array(img))
        bboxes.append(np.array(bbox))
    imgs = torch.from_numpy(np.array(imgs).transpose((0, 3, 1, 2))).float()
    bboxes = np.array(bboxes)
    return imgs, bboxes, gt


class cocoDataSet(Dataset):
    num_classes = 50
    defult_resolution = [243, 243]

    def __init__(self, root_dir, split, resize=(243, 243), pretrain=False):
        """
            @split: val or train
        """
        self.root_dir = root_dir
        self.resize = resize
        self.pretrain = pretrain
        self.anno_path = os.path.join(root_dir, "fewshotlogodetection_round1_{}_202204", "{}", "annotations",
                                      'instances_{}2017.json').format(split, split, split)
        self.imgDir_path = os.path.join(root_dir, "fewshotlogodetection_round1_{}_202204", "{}", "images").format(split,
                                                                                                                  split)
        self._coco = coco.COCO(annotation_file=self.anno_path)
        self._ids = list(sorted(self._coco.imgs.keys()))
        self.coco_class = dict([(v["id"], v["name"]) for k, v in self._coco.cats.items()])

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, item):
        # get all annotations information of picture
        annos = self._coco.getAnnIds(imgIds=self._ids[item])
        label_info = self._coco.loadAnns(annos)
        img_Info = self._coco.loadImgs(self._ids[item])

        img_abs_path = os.path.join(self.imgDir_path, img_Info[0]["file_name"])
        img = Image.open(img_abs_path).convert("RGB")
        # img = cv2.imread(img_abs_path)
        bboxAndlabel = self._get_format_bbox(label_info)
        # resize
        if self.resize:
            img, bboxAndlabel = self._img_label_transform(img, bboxAndlabel, self.resize[0], self.resize[1])
            gt = self._get_gt(bboxAndlabel)
        else:
            gt = 0
        return img, bboxAndlabel, gt

    def _img_label_transform(self, img, bbox, W_size, H_size):
        """
            @param x_size: after transform size of x
            @param y_size: after transform size of y
            @return: img and label after transform
        """
        new_bbox = []
        W, H = img.size
        scale_W, scale_H = W_size / float(W), H_size / float(H)
        for box in bbox:
            b1, b2, b3, b4, label = box
            b1, b3 = b1 * scale_W, b3 * scale_W
            b2, b4 = b2 * scale_H, b4 * scale_H
            new_bbox.append(np.array([b1, b2, b3, b4, label]))
        new_bbox = np.array(new_bbox)
        img = T.Resize((W_size, H_size))(img)
        return img, new_bbox

    def _get_format_bbox(self, label_info):
        """
            @label_info: label_info: segmentation information, include bbox, category id
            example:label_info: [{'id': 2528, 'image_id': 1105, 'category_id': 32, 'segmentation': [[783, 25, 783, 56.0,
             783, 87, 878.0, 87, 973, 87, 973, 56.0, 973, 25, 878.0, 25]], 'bbox': [783, 25, 190, 62], 'iscrowd': 0,
             'area': 12033}]
        """
        bbox_lst = []
        for item in label_info:
            category_id = item['category_id']
            box = item["bbox"]
            format_box = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2, box[2] / 2, box[3] / 2, category_id])
            bbox_lst.append(format_box)
        return np.array(bbox_lst)

    def _get_gt(self, bbox):
        gt_maker = centerNetGT(self.resize)
        gt = gt_maker(bbox)
        return gt

    def draw_bbox(self, img_id, root_path, label_info, text=None):
        """
        label_info: segmentation information, include bbox
        """
        # img information, file_name
        img_Info = self._coco.loadImgs(img_id)
        img_abs_dir = os.path.join(root_path, img_Info[0]["file_name"])
        img = cv2.imread(img_abs_dir)
        bbox_lst = self._get_format_bbox(label_info)
        for (box, cat_id) in bbox_lst:
            text_info = self.coco_class[cat_id]
            if text is not None:
                text_info += text
            text_len = len(text_info)
            x1, y1, x2, y2 = [int(box[0] - box[2]), int(box[1] - box[3]), int(box[0] + box[2]), int(box[1] + box[3])]
            img = cv2.rectangle(img, (x1, y1),
                                (x2, y2), (255, 0, 0), 1)
            img = cv2.rectangle(img, (int(box[0] - box[2]), int(box[1] - box[3]) - 16),
                                (int(box[0] - box[2]) + text_len * 11, int(box[1] - box[3])), (250, 250, 20), -1)
            img = cv2.putText(img, text_info, (int(box[0] - box[2]), int(box[1] - box[3]) - 5),
                              cv2.FONT_HERSHEY_COMPLEX,
                              0.5, (0, 0, 0))
        cv2.imshow(str(img_Info[0]["id"]), img)
        cv2.waitKey(0)


class coco_dataloader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 collate_fn=collate, resize=(243, 243)):
        if training:
            split = "train"
        else:
            split = "val"
        self.data_set = cocoDataSet(data_dir, split, resize=resize)
        super().__init__(self.data_set, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
