import os
import cv2
import numpy as np
import pycocotools.coco as coco


def _get_format_bbox(target_lst):
    bbox_lst = []
    for item in target_lst:
        box = item["bbox"]
        format_box = (box[0] + box[2] / 2, box[1] + box[3] / 2, box[2] / 2, box[3] / 2)
        bbox_lst.append(format_box)
    return bbox_lst


path = f"/mnt/data/coco2017/annotations/instances_val2017.json"
data = coco.COCO(annotation_file=path)
ids = list(sorted(data.imgs.keys()))

coco_class = dict([(v["id"], v["name"]) for k, v in data.cats.items()])
print(coco_class)

for img_id in ids[7:10]:
    # get annotation info
    ann_ids = data.getAnnIds(imgIds=img_id)
    targets = data.loadAnns(ann_ids)
    print(targets)
    lst = _get_format_bbox(targets)
    # print(lst)
    img_Info = data.loadImgs(img_id)
    # print(img_Info)
    img_abs_dir = os.path.join(f"/mnt/data/coco2017/val2017", img_Info[0]["file_name"])
    img = cv2.imread(img_abs_dir)
    for box in lst:
        x1, y1, x2, y2 = int(box[0] - box[2]), int(box[1] - box[3]), int(box[0] + box[2]), int(box[1] + box[3])
        img = cv2.rectangle(img, (x1, y1),
                            (x2, y2), (255, 0, 0), 1)
    cv2.imshow(str(img_Info[0]["id"]), img)
    cv2.waitKey(0)
