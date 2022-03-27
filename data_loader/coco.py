import os
import pycocotools.coco as coco

from base import BaseDataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class cocoDataSet(Dataset):
    num_classes = 80
    defult_resolution = [512, 512]

    def __init__(self, root_dir, split):
        self.anno_path = os.path.join(root_dir, "annotations", 'instances_{}2017.json').format(split)
        self.imgDir_path = os.path.join(root_dir, "{}2017").format(split)
        self._coco = coco.COCO(annotation_file=self.anno_path)
        self._ids = list(sorted(self._coco.imgs.keys()))

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, item):
        # get all annotations information of picture
        annos = coco.COCO.getAnnIds(imgIds=self._ids[item])
        target_dict_list = self._coco.loadAnns(annos)

    def _get_format_bbox(self, target_lst):
        bbox_lst = []
        for item in target_lst:
            box = item["bbox"]
            format_box = (box[0] + box[2] / 2, box[1] + box[3] / 2, box[2] / 2, box[3] / 2)
            bbox_lst.append(format_box)
        return bbox_lst
