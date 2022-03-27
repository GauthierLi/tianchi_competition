"""
    @author: gauthierLi
    @date:03/07/2022
"""
import os
import cv2
import pycocotools.coco as coco

from base import BaseDataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class cocoDataSet(Dataset):
    num_classes = 80
    defult_resolution = [512, 512]

    def __init__(self, root_dir, split):
        """
            @split: val or train
        """
        self.anno_path = os.path.join(root_dir, "annotations", 'instances_{}2017.json').format(split)
        self.imgDir_path = os.path.join(root_dir, "{}2017").format(split)
        self._coco = coco.COCO(annotation_file=self.anno_path)
        self._ids = list(sorted(self._coco.imgs.keys()))
        self.coco_class = dict([(v["id"], v["name"]) for k, v in self._coco.cats.items()])

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, item):
        # get all annotations information of picture
        annos = coco.COCO.getAnnIds(imgIds=self._ids[item])
        label_info = self._coco.loadAnns(annos)
        img_Info = self._coco.loadImgs(self._ids[item])

        img_abs_path = os.path.join(self.imgDir_path, img_Info[0]["file_name"])
        img = cv2.imread(img_abs_path)
        bbox, label = self._get_format_bbox(label_info)

        # gt = self._get_gt(bbox, label)
        return img, (bbox, label)


    def _get_format_bbox(self, label_info):
        """
            @label_info: label_info: segmentation information, include bbox, category id
        """
        bbox_lst = []
        for item in label_info:
            category_id = item['category_id']
            box = item["bbox"]
            format_box = [(box[0] + box[2] / 2, box[1] + box[3] / 2, box[2] / 2, box[3] / 2), category_id]
            bbox_lst.append(format_box)
        return bbox_lst

    def _get_gt(self, bbox, label):
        pass

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
