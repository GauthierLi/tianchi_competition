import cv2
import torch
import numpy as np

from copy import deepcopy
from utils.centerNetGT import centerNetGT
from model.model import detector
from data_loader.coco import coco_dataloader
from test_area.testArea import draw_bbox

config = r"/media/gauthierli-org/GauLi/code/tainchi_competition/saved/models/kernel_generator/0422_150723/config.json"
resume = r"/media/gauthierli-org/GauLi/code/tainchi_competition/saved/models/kernel_generator/0422_150723/model_best.pth"
reference = r"/media/gauthierli-org/GauLi/code/tainchi_competition/test_area/logo_imgs"
tst_detector = detector(config, resume, reference)
checkpoint = torch.load(
    r"/media/gauthierli-org/GauLi/code/tainchi_competition/saved/models/detect network/0503_221846/model_best.pth")
state_dict = checkpoint['state_dict']
tst_detector.load_state_dict(state_dict)

device = torch.device("cuda")
tst_detector.to(device)
tst_detector.eval()


dataLoader = coco_dataloader(f"/home/gauthierli-org/data/data/fewshot", batch_size=1, training=True)
with torch.no_grad():
    for img, bboxAndlabel, gt in dataLoader:
        print(bboxAndlabel.shape)
        label = bboxAndlabel[0][0][-1].astype(int)

        cv_img = img.squeeze().cpu().numpy().astype(np.uint8).transpose((1, 2, 0))[:, :, ::-1]
        # cv2.imshow("img", cv_img)
        # cv2.waitKey(0)

        img = img.to(device)
        result = tst_detector(img).cpu()

        bbox = centerNetGT.parse_to_standard(1, result[0])

        for i, fram in enumerate(result[0]):
            fram = fram.numpy()
            print(fram.tolist())
            cv2.imshow("img" + str(i), fram)
            cv2.waitKey(0)

        cv_img = cv_img.copy()
        draw_bbox(cv_img, bbox)
