"""
    @author:gauthierLi
    @data:03/27/2022
    @func:test only
"""
import json
import os
import cv2
import csv
import torch
import numpy as np
from PIL import Image
import pycocotools.coco as coco

from model.model import kernel_extract_network
import torch.nn.functional as F
from data_loader.data_loaders import MnistDataLoader
from data_loader.coco import coco_dataloader
from data_loader.pretrain_coco import pretrain_coco_dataloader
from data_loader.branch_data import *
from utils.centerNetGT import centerNetGT

from model.model import kernel_extract_network, kernel_generator, classify_decoder
from torchvision.models import convnext_base


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
        print(boxandlabel)
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
        f"/home/gauthierli-org/data/data/fewshot", batch_size=2,
        training=True)
    for img, bboxAndlabel, gt in dataLoader:
        print("img shape", img.size())
        print("bbox shape", bboxAndlabel.shape)
        print("gt shape", gt.size(), type(gt))
        # img = img.squeeze().numpy().astype(np.uint8).transpose((1, 2, 0))[:, :, ::-1]
        # img = np.ascontiguousarray(img)
        # # cv2.imshow("img", img)
        # # cv2.waitKey()
        # label = int(bboxAndlabel[0][0][-1])
        # center = gt.numpy()[label - 1]
        # w = gt.numpy()[-4]
        # print("org", bboxAndlabel)
        # # back = centerNetGT.parse_to_standard(0, gt)
        # # print("back", back)
        # cv2.imshow("center", center)
        # cv2.imshow("w", w)
        # draw_bbox(img, bboxAndlabel[0])
        # # draw_bbox(img, back)


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


def tst_cocotools():
    annos_path = r"/home/gauthierli-org/data/data/fewshot/fewshotlogodetection_round1_{}_202204/{}/annotations/instances_{}2017.json".format(
        "train", "train", "train")
    data = coco.COCO(annotation_file=annos_path)
    coco_class = dict([(v["id"], v["name"]) for k, v in data.cats.items()])
    category_id = list(coco_class.keys())
    print(data.anns[56])
    print(list(data.anns.items())[56])

    category_slice_dict = {}
    for category_id in category_id:
        category_slice_dict[category_id] = []
        imgs = data.getImgIds(catIds=category_id)
        for img_id in imgs:
            annos = data.getAnnIds(imgIds=img_id)
            # print("annos: ", annos, "\n")
            label_info = data.loadAnns(annos)
            # print("label_info: ", label_info, "\n")
            img_info = data.loadImgs(img_id)
            # print("img_info ", img_info, "\n")
            for item in label_info:
                tmp_dict = {}
                tmp_dict['id'] = item['id']
                tmp_dict['file_name'] = img_info[0]["file_name"]
                tmp_dict['bbox'] = item['bbox']
                tmp_dict['category_id'] = category_id
                category_slice_dict[category_id].append(tmp_dict)

    file_handle = open(r"test_area/category_slice_dict.json", "w")
    json.dump(category_slice_dict, file_handle, indent=4)
    file_handle.close()

    # annos = data.getAnnIds(imgIds=ids[0])
    # label_info = data.loadAnns(annos)
    # img_Info = data.loadImgs(ids[0])
    # print(label_info, "\n \n", img_Info)


def resize_img(path, size=(243, 243)):
    img = cv2.imread(path)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(dim=0)
    resize = torchvision.transforms.Resize((243, 243))
    img = resize(img)
    return img


def tst_pretrain_coco():
    root = r"/home/gauthierli-org/data/data/fewshot"
    loader = pretrain_coco_dataloader(root=root, batch_size=1)
    for img, label in loader:
        img = img.squeeze().numpy().astype(np.uint8).transpose((1, 2, 0))[:, :, ::-1]
        cv2.imshow(f"img {label}", img)
        print(label)
        cv2.waitKey()
        cv2.destroyWindow(f"img {label}")


if __name__ == "__main__":
    # tst_backbone()
    # tst_MNist_dataLoader()
    tst_dataLoader()
    # tst_PIL()
    # tst_csv_reader()
    # tst_branch_dataloader()
    # tst_torch_tensor()
    # tst_cocotools()
    # tst_pretrain_coco()
