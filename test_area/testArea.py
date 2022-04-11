"""
    @author:gauthierLi
    @data:03/27/2022
    @func:test only
"""
import os
import cv2
import csv
import torch
import numpy as np
from PIL import Image
import pycocotools.coco as coco
import torch.nn.functional as F
from data_loader.data_loaders import MnistDataLoader
from data_loader.coco import coco_dataloader
from data_loader.branch_data import *

from model.model import kernel_extract_network, kernel_generator, classify_decoder


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
        box = boxandlabel[:-1]
        text_info = str(boxandlabel[-1])
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


def tst_dataLoader():
    device = torch.device("cuda")
    dataLoader = coco_dataloader(
        f"/home/gauthierli-org/data/data/fewshot", batch_size=1,
        training=True, resize=False)
    for img, bboxAndlabel in dataLoader:
        print("img shape", img.shape)
        print("bbox shape", bboxAndlabel.shape)
        img = img.squeeze().numpy().astype(np.uint8).transpose((1, 2, 0))[:, :, ::-1]
        img = np.ascontiguousarray(img)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        draw_bbox(img, bboxAndlabel[0])


def tst_MNist_dataLoader():
    data_loader = MnistDataLoader("data/", 2)
    for data, label in data_loader:
        print(data)
        print(label)
        break


def tst_backbone():
    device = torch.device("cuda")
    a = torch.ones([2, 3, 243, 243]).to(device)
    tst_kernel_extract = kernel_extract_network()
    tst_kernel_extract.to(device)
    print(tst_kernel_extract(a).size())
    lst = tst_kernel_extract(a)[0].cpu().data.numpy()
    print("total class {}, after softmax is {}".format(len(lst), lst.sum()))

    cg_encoder = kernel_generator(inchannel=3, stride=1, pad=1)
    cg_encoder.to(device)
    cg_encoder.eval()
    print(cg_encoder(a).size())


def tst_PIL():
    img_path = r"/home/gauthierli-org/data/data/coco/val2017/000000000776.jpg"
    img = Image.open(img_path)
    print(img.size)


def tst_csv_reader():
    path = r"/home/gauthierli-org/data/data/vehicle-logos-dataset/structure.csv"
    csv_file = open(path, "r")
    data_list_dict = list(csv.DictReader(csv_file))
    KV_map = {}
    parse_map = {}
    v = 0
    for ele in data_list_dict:
        if ele["Template Name"] not in KV_map.keys():
            KV_map[ele["Template Name"]] = v
            parse_map[v] = ele["Template Name"]
            v += 1

    print(KV_map)
    print("\n")
    print(parse_map)


def tst_branch_dataloader():
    device = torch.device("cuda")
    root = r"/home/gauthierli-org/data/data/vehicle-logos-dataset/"
    train_loader = branch_data_loader(root, 2, True, 0.2)
    val_loader = train_loader.split_validation()
    tst_kernel_extract = kernel_extract_network()
    tst_kernel_extract.to(device)
    for img, label in val_loader:
        print(img.size())
        print(label.size(), inverse_dict[label.data.numpy().argmax()])
        result = tst_kernel_extract(img.to(device))
        print(F.cross_entropy(result, label.to(device)))


def tst_torch_tensor():
    ten1 = torch.tensor([[9, 8, 7]])
    ten2 = torch.tensor([[7, 5, 2]])
    print(ten1 == ten2)


if __name__ == "__main__":
    # tst_backbone()
    # tst_MNist_dataLoader()
    tst_dataLoader()
    # tst_PIL()
    # tst_csv_reader()
    # tst_branch_dataloader()
    # tst_torch_tensor()
