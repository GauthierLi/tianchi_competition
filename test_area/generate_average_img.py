import os

import cv2
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F

root = r"/home/gauthierli-org/data/data/vehicle-logos-dataset"
src_img_dict = json.load(open("src_img.json", "r"))
labels = src_img_dict.keys()

"""
labels:
['Toyota', 'Mitsubishi', 'Seat', 'Suzuki', 'Opel', 'Honda 1', 'Renault', 'Mercedes 1', 'Volkswagen', 'Subaru',
 'Dacia 1', 'Citroen', 'Land Rover', 'Tesla', 'Mazda', 'Nissan', 'Hyundai', 'Lancia 1', 'Skoda 1', 'Chevrolet 2',
 'Porsche', 'Peugeot', 'Jeep', 'Mini', 'Kia 1', 'Lexus', 'Smart 2', 'Volvo 1', 'GMC', 'Ford', 'Daewoo 1', 'Acura',
 'Alfa Romeo', 'BMW']
 each class contains 16 imgs
"""


class format_imgs(nn.Module):
    def __init__(self):
        super(format_imgs, self).__init__()
        self.trans = nn.Sequential(torchvision.transforms.Resize((243, 243)))

    def forward(self, x):
        return self.trans(x)


def generate_average_img():
    r"""
    generate a average picture which close to all samples in same class
    :return:
    """
    for branch in list(labels):
        device = torch.device("cuda")
        objection = torch.ones((1, 243, 243), dtype=torch.float64, requires_grad=True)
        objection1 = objection.to(device)

        for i in range(40):
            if i % 40 == 0:
                print("\r current {} / total {}".format(i, 200), end="")
            for img_path in src_img_dict[branch]:
                img = cv2.imread(os.path.join(root, img_path), cv2.IMREAD_GRAYSCALE).astype(float)
                img = torch.from_numpy(img)
                img = img.unsqueeze(dim=0)
                img = format_imgs()(img)

                img1 = img.to(device)

                loss = nn.MSELoss()(img1, objection1)
                loss.backward()
                objection.data = objection.data - 0.1 * objection.grad.data

        objection = objection.cpu().data.numpy().transpose((1, 2, 0))
        cv2.imwrite(os.path.join("logo_imgs", branch + "_GRAY.jpg"), objection)
        print(objection.shape)
        cv2.imshow("img " + branch, objection)
        cv2.waitKey(1)


def walk_imgs():
    path_list = list(os.walk("/home/gauthierli-org/code/testArea/logo_imgs/color_label"))
    for img_path in path_list[0][2]:
        img = cv2.imread("logo_imgs/color_label/" + img_path)
        cv2.imshow(img_path, img)
        cv2.waitKey(0)


if __name__ == "__main__":
    generate_average_img()