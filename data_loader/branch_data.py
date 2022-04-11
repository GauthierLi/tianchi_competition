"""
    used on vehicle logo dataset
    @author:gauthierLi
    @date:04/05/2022
"""
import os
import cv2
import csv
import numpy as np
import torchvision.transforms
from PIL import Image
import torch
import torchvision.transforms as T

from base import BaseDataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

mapping_dict = {'Toyota': 0, 'Mitsubishi': 1, 'Seat': 2, 'Suzuki': 3, 'Opel': 4, 'Honda 1': 5, 'Renault': 6,
                'Mercedes 1': 7, 'Volkswagen': 8, 'Subaru': 9, 'Dacia 1': 10, 'Citroen': 11, 'Land Rover': 12,
                'Tesla': 13, 'Mazda': 14, 'Nissan': 15, 'Hyundai': 16, 'Lancia 1': 17, 'Skoda 1': 18, 'Chevrolet 2': 19,
                'Porsche': 20, 'Peugeot': 21, 'Jeep': 22, 'Mini': 23, 'Kia 1': 24, 'Lexus': 25, 'Smart 2': 26,
                'Volvo 1': 27, 'GMC': 28, 'Ford': 29, 'Daewoo 1': 30, 'Acura': 31, 'Alfa Romeo': 32, 'BMW': 33}

inverse_dict = {0: 'Toyota', 1: 'Mitsubishi', 2: 'Seat', 3: 'Suzuki', 4: 'Opel', 5: 'Honda 1', 6: 'Renault',
                7: 'Mercedes 1', 8: 'Volkswagen', 9: 'Subaru', 10: 'Dacia 1', 11: 'Citroen', 12: 'Land Rover',
                13: 'Tesla', 14: 'Mazda', 15: 'Nissan', 16: 'Hyundai', 17: 'Lancia 1', 18: 'Skoda 1', 19: 'Chevrolet 2',
                20: 'Porsche', 21: 'Peugeot', 22: 'Jeep', 23: 'Mini', 24: 'Kia 1', 25: 'Lexus', 26: 'Smart 2',
                27: 'Volvo 1', 28: 'GMC', 29: 'Ford', 30: 'Daewoo 1', 31: 'Acura', 32: 'Alfa Romeo', 33: 'BMW'}


class branch_dataset(Dataset):
    def __init__(self, root):
        self.root = root
        data_csv_file = open(os.path.join(self.root, "structure.csv"), "r")
        # Class,Template Name,Image,Mask,Resolution
        self.data_dict = list(csv.DictReader(data_csv_file))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data_item = self.data_dict[item]
        img_path = os.path.join(self.root, data_item["Image"])
        img_label = mapping_dict[data_item["Template Name"]]
        img_label = self._one_hot(img_label)

        img = Image.open(img_path)
        img = torch.from_numpy(np.array(T.Resize((243, 243))(img)).transpose((2, 0, 1))).float()
        return img, img_label

    def _one_hot(self, x):
        result = torch.zeros(50)
        result[x] = 1
        return result

class branch_data_loader(BaseDataLoader):
    def __init__(self, root, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 collate_fn=default_collate):
        self.data_set = branch_dataset(root)
        super(branch_data_loader, self).__init__(self.data_set, batch_size, shuffle, validation_split, num_workers,
                                                 collate_fn=collate_fn)

