"""
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

mapping_dict = {'Toyota': 0, 'Mitsubishi': 1, 'Seat': 2, 'Suzuki': 4, 'Opel': 5, 'Honda 1': 6, 'Renault': 32,
                'Mercedes 1': 33, 'Volkswagen': 35, 'Subaru': 105, 'Dacia 1': 159, 'Citroen': 163, 'Land Rover': 192,
                'Tesla': 193, 'Mazda': 194, 'Nissan': 195, 'Hyundai': 196, 'Lancia 1': 197, 'Skoda 1': 198,
                'Chevrolet 2': 199, 'Porsche': 202, 'Peugeot': 205, 'Jeep': 211, 'Mini': 217, 'Kia 1': 219,
                'Lexus': 229, 'Smart 2': 260, 'Volvo 1': 309, 'GMC': 310, 'Ford': 317, 'Daewoo 1': 349, 'Acura': 378,
                'Alfa Romeo': 402, 'BMW': 484}

inverse_dict = {0: 'Toyota', 1: 'Mitsubishi', 2: 'Seat', 4: 'Suzuki', 5: 'Opel', 6: 'Honda 1', 32: 'Renault',
                33: 'Mercedes 1', 35: 'Volkswagen', 105: 'Subaru', 159: 'Dacia 1', 163: 'Citroen', 192: 'Land Rover',
                193: 'Tesla', 194: 'Mazda', 195: 'Nissan', 196: 'Hyundai', 197: 'Lancia 1', 198: 'Skoda 1',
                199: 'Chevrolet 2', 202: 'Porsche', 205: 'Peugeot', 211: 'Jeep', 217: 'Mini', 219: 'Kia 1',
                229: 'Lexus', 260: 'Smart 2', 309: 'Volvo 1', 310: 'GMC', 317: 'Ford', 349: 'Daewoo 1', 378: 'Acura',
                402: 'Alfa Romeo', 484: 'BMW'}


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

        img = Image.open(img_path)
        img = torch.from_numpy(np.array(T.Resize((243, 243))(img)))
        return img, img_label


class branch_data_loader(BaseDataLoader):
    def __init__(self, root, batch_size, shuffle=True, validation_split=0.0, num_workers=1,
                 collate_fn=default_collate):
        self.data_set = branch_dataset(root)
        super(branch_data_loader, self).__init__(self.data_set, batch_size, shuffle, validation_split, num_workers,
                                                 collate_fn=collate_fn)

