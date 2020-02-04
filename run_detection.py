import torch
import sys
from detection.detection_model import Detection
from torch.utils.data import DataLoader
import cv2
import os

import datasets

def draw_bbox(imgName, bboxes, scores, writeFolder):
    img = cv2.imread(imgName)
    import math
    for bbox, score in zip(bboxes, scores):
        xmin, ymin, xmax, ymax = math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)
        cv2.putText(img, text=str(score), org=(xmin - 10, ymin - 5), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join(writeFolder, imgName.split('/')[-1]), img)


def run_detection(loader, source_folder, backbone='maskrcnn'):
    model = Detection(backbone=backbone).to(0).eval()
    for batch_image, batch_bbox, batch_class, batch_name in loader:
        # print(batch_image.shape)
        batch_out = model(list(batch_image.cuda()))
        boxes = []
        scores = []
        idxss = []
        for out in batch_out:
            boxes.append(out['boxes'].cpu().detach().numpy())
            scores.append(out['scores'].cpu().detach().numpy())
            idxs = model.nms_postprocess(bboxs=boxes[-1], scores=scores[-1])
            idxss.append(idxs)
        print(batch_name)
        for box, score, name, idxs in zip(boxes, scores, batch_name, idxss):
            image_name = os.path.join(source_folder, name)
            print(idxs)
            draw_bbox(image_name, box[idxs] if idxs.shape[0] > 0 else [], score[idxs] if idxs.shape[0] > 0 else [], write_folder)


if __name__ == '__main__':
    dataset = sys.argv[1]
    source_folder = sys.argv[2]
    write_folder = sys.argv[3]
    if dataset == 'VOC2012':
        # loader = DataLoader(dataset=datasets.VOCDataset, batch_size=2, shuffle=False, num_workers=2)
        ds = datasets.VOCDataset()
    else:
        ds = datasets.CocoDataset()
    loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=2)
    run_detection(loader, source_folder)
