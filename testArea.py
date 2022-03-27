import os
import cv2
import numpy as np
import pycocotools.coco as coco


def _get_format_bbox(label_info):
    bbox_lst = []
    for item in label_info:
        category_id = item['category_id']
        box = item["bbox"]
        format_box = [(box[0] + box[2] / 2, box[1] + box[3] / 2, box[2] / 2, box[3] / 2), category_id]
        bbox_lst.append(format_box)
    return bbox_lst


def draw_bbox(img_id, root_path, label_info, text=None):
    """
    label_info: segmentation information, include bbox
    """
    # img information, file_name
    img_Info = data.loadImgs(img_id)
    img_abs_dir = os.path.join(root_path, img_Info[0]["file_name"])
    img = cv2.imread(img_abs_dir)
    bbox_lst = _get_format_bbox(label_info)
    for (box, cat_id) in bbox_lst:
        text_info = coco_class[cat_id]
        if text is not None:
            text_info += text
        text_len = len(text_info)
        x1, y1, x2, y2 = [int(box[0] - box[2]), int(box[1] - box[3]), int(box[0] + box[2]), int(box[1] + box[3])]
        img = cv2.rectangle(img, (x1, y1),
                            (x2, y2), (255, 0, 0), 1)
        img = cv2.rectangle(img, (int(box[0] - box[2]), int(box[1] - box[3]) - 16),
                            (int(box[0] - box[2]) + text_len * 11, int(box[1] - box[3])), (250, 250, 20), -1)
        img = cv2.putText(img, text_info, (int(box[0] - box[2]), int(box[1] - box[3]) - 5), cv2.FONT_HERSHEY_COMPLEX,
                          0.5, (0, 0, 0))
    cv2.imshow(str(img_Info[0]["id"]), img)
    cv2.waitKey(0)


if __name__ == "__main__":
    path = f"/mnt/data/coco2017/annotations/instances_val2017.json"
    data = coco.COCO(annotation_file=path)
    ids = list(sorted(data.imgs.keys()))

    coco_class = dict([(v["id"], v["name"]) for k, v in data.cats.items()])
    print(coco_class)

    for img_id in ids[7:10]:
        # get annotation info
        ann_ids = data.getAnnIds(imgIds=img_id)
        print(ann_ids)  # [301431, 559508, 560228, 633972] four targets
        # segmentation information, include bbox
        label_info = data.loadAnns(ann_ids)
        # img information , include filename,'height' and 'width'
        img_Info = data.loadImgs(img_id)

        bbox_lst = _get_format_bbox(label_info)

        draw_bbox(img_id, f"/mnt/data/coco2017/val2017", label_info)