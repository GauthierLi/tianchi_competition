import os

import cv2
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import PIL.Image as Image

root = r"/home/gauthierli-org/data/data/fewshot/fewshotlogodetection_round1_train_202204/train/images"
src_img_dict = json.load(open("category_slice_dict.json", "r"))
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
    epoch = 25
    for label in list(labels):
        device = torch.device("cuda")
        objection = torch.ones((3, 243, 243), dtype=torch.float64, requires_grad=True)
        objection1 = objection.to(device)

        for i in range(epoch):
            if i % 8 == 0:
                print("\r label{}, current {} epoch/ total {}".format(label, i, epoch), end="")
            for img_info in src_img_dict[label]:
                img = Image.open(os.path.join(root, img_info["file_name"])).convert("RGB")
                img = np.array(img).astype(np.float64)
                x, y, w, h = img_info['bbox']
                img = img[y:y + h, x:x + w, :]
                img = img.transpose((2, 0, 1))

                # img = img.astype(np.uint8).transpose((1, 2, 0))[:, :, :: -1]
                # cv2.imshow(f"img{label}", img)
                # cv2.waitKey()

                img = torch.from_numpy(img)
                img = format_imgs()(img)

                # imgs = img.data.numpy().transpose((1, 2, 0)).astype(np.uint8)[:,:,::-1]
                # cv2.imshow("img", imgs)
                # cv2.waitKey()

                img1 = img.to(device)

                loss = nn.MSELoss()(img1, objection1)
                loss.backward()
                objection.data = objection.data - 0.1 * objection.grad.data

        objection = objection.cpu().data.numpy().transpose((1, 2, 0))
        objection = objection.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join("logo_imgs", label + "_RGB.jpg"), objection)
        print(objection.shape)
        cv2.imshow("img " + label, objection)
        cv2.waitKey(1)
        cv2.destroyWindow("img " + label)


def walk_imgs():
    path_list = list(os.walk("/home/gauthierli-org/code/testArea/logo_imgs/color_label"))
    for img_path in path_list[0][2]:
        img = cv2.imread("logo_imgs/color_label/" + img_path)
        cv2.imshow(img_path, img)
        cv2.waitKey(0)


if __name__ == "__main__":
    generate_average_img()
