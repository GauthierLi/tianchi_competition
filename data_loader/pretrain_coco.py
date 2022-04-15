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
from torch.utils.data.dataloader import default_collate

query_dict = {1: '冰墩墩', 2: 'Sanyo/三洋', 3: 'Eifini/伊芙丽', 4: 'PSALTER/诗篇', 5: 'Beaster', 6: 'ON/昂跑', 7: 'BYREDO/柏芮朵',
              8: 'Ubras', 9: 'Eternelle', 10: 'PERFECT DIARY/完美日记', 11: '花西子', 12: 'Clarins/娇韵诗', 13: "L'occitane/欧舒丹",
              14: 'Versace/范思哲', 15: 'Mizuno/美津浓', 16: 'Lining/李宁', 17: 'DOUBLE STAR/双星', 18: 'YONEX/尤尼克斯',
              19: 'Tory Burch/汤丽柏琦', 20: 'Gucci/古驰', 21: 'Louis Vuitton/路易威登', 22: 'CARTELO/卡帝乐鳄鱼', 23: 'JORDAN',
              24: 'KENZO', 25: 'UNDEFEATED', 26: 'BOY LONDON', 27: 'TREYO/雀友', 28: 'carhartt', 29: '洁柔',
              30: 'Blancpain/宝珀', 31: 'GXG', 32: '乐町', 33: 'Diadora/迪亚多纳', 34: 'TUCANO/啄木鸟', 35: 'Loewe',
              36: 'Granite Gear', 37: 'DESCENTE/迪桑特', 38: 'OSPREY', 39: 'Swatch/斯沃琪', 40: 'erke/鸿星尔克',
              41: 'Massimo Dutti', 42: 'PINKO', 43: 'PALLADIUM', 44: 'origins/悦木之源', 45: 'Trendiano', 46: '音儿',
              47: 'Monster Guardians', 48: '敷尔佳', 49: 'IPSA/茵芙莎', 50: 'Schwarzkopf/施华蔻'}

inverse_query_dict = {'冰墩墩': 1, 'Sanyo/三洋': 2, 'Eifini/伊芙丽': 3, 'PSALTER/诗篇': 4, 'Beaster': 5, 'ON/昂跑': 6,
                      'BYREDO/柏芮朵': 7, 'Ubras': 8, 'Eternelle': 9, 'PERFECT DIARY/完美日记': 10, '花西子': 11,
                      'Clarins/娇韵诗': 12, "L'occitane/欧舒丹": 13, 'Versace/范思哲': 14, 'Mizuno/美津浓': 15, 'Lining/李宁': 16,
                      'DOUBLE STAR/双星': 17, 'YONEX/尤尼克斯': 18, 'Tory Burch/汤丽柏琦': 19, 'Gucci/古驰': 20,
                      'Louis Vuitton/路易威登': 21, 'CARTELO/卡帝乐鳄鱼': 22, 'JORDAN': 23, 'KENZO': 24, 'UNDEFEATED': 25,
                      'BOY LONDON': 26, 'TREYO/雀友': 27, 'carhartt': 28, '洁柔': 29, 'Blancpain/宝珀': 30, 'GXG': 31,
                      '乐町': 32, 'Diadora/迪亚多纳': 33, 'TUCANO/啄木鸟': 34, 'Loewe': 35, 'Granite Gear': 36,
                      'DESCENTE/迪桑特': 37, 'OSPREY': 38, 'Swatch/斯沃琪': 39, 'erke/鸿星尔克': 40, 'Massimo Dutti': 41,
                      'PINKO': 42, 'PALLADIUM': 43, 'origins/悦木之源': 44, 'Trendiano': 45, '音儿': 46,
                      'Monster Guardians': 47, '敷尔佳': 48, 'IPSA/茵芙莎': 49, 'Schwarzkopf/施华蔻': 50}


class pretrain_cocoDataSet(Dataset):
    num_classes = 50
    defult_resolution = [243, 243]

    def __init__(self, root_dir, split, resize=(243, 243)):
        """
            @split: val or train
        """
        self.root_dir = root_dir
        self.resize = resize
        self.anno_path = os.path.join(root_dir, "fewshotlogodetection_round1_{}_202204", "{}", "annotations",
                                      'instances_{}2017.json').format(split, split, split)
        self.imgDir_path = os.path.join(root_dir, "fewshotlogodetection_round1_{}_202204", "{}", "images").format(split,
                                                                                                                  split)
        self._coco = coco.COCO(annotation_file=self.anno_path)
        self._ids = list(sorted(self._coco.anns.keys()))
        self.coco_class = dict([(v["id"], v["name"]) for k, v in self._coco.cats.items()])

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, item):
        # get all annotations information of picture
        annos = self._coco.anns[item]
        bbox = annos['bbox']
        label = annos['category_id']
        label = self._one_hot(label)
        img_name = self._coco.loadImgs(annos['image_id'])[0]['file_name']

        img_abs_path = os.path.join(self.imgDir_path, img_name)
        img = Image.open(img_abs_path).convert("RGB")

        img = np.array(img)
        img = img[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]), :]
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img)


        img = torchvision.transforms.Resize(self.resize)(img)

        return img, label

    def _one_hot(self, x):
        result = torch.zeros(50)
        result[x-1] = 1
        return result


class pretrain_coco_dataloader(BaseDataLoader):
    def __init__(self, root, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 collate_fn=default_collate, resize=(243, 243)):
        if training:
            split = "train"
        else:
            split = "val"
        self.data_set = pretrain_cocoDataSet(root, split, resize=resize)
        super().__init__(self.data_set, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
