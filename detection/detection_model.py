import torch.nn as nn
import numpy as np

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class Detection(nn.Module):
    def __init__(self, backbone='fasterrcnn', use_pretrained=True, num_classes=91):
        super(Detection, self).__init__()
        if backbone == 'fasterrcnn':
            self.detection_backbone = fasterrcnn_resnet50_fpn(pretrained=use_pretrained, num_classes=num_classes).cuda()
        else:
            self.detection_backbone = maskrcnn_resnet50_fpn(pretrained=use_pretrained, num_classes=num_classes).cuda()

    def forward(self, X, target=None):
        if self.training:
            out = self.detection_backbone(X, target)
        else:
            out = self.detection_backbone(X)
        return out

    def nms_postprocess(self, bboxs, scores, threshold=0.5):
        '''
        NMS implementation. Pure Python Version
        :param bboxs: bounding boxes to filter
        :param threshold: threshold for filtering bbox
        :return: filtered bboxes
        '''
        x1 = bboxs[:, 0]
        y1 = bboxs[:, 1]
        x2 = bboxs[:, 2]
        y2 = bboxs[:, 3]
        areas = (x2 - x1 + 0.01) * (y2 - y1 + 0.01)

        order = scores.argsort()[::-1]
        rest = []

        while order.size > 0:
            i = order[0]
            rest.append(i)

            # get intersection coordinates
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.maximum(x2[i], x2[order[1:]])
            yy2 = np.maximum(y2[i], y2[order[1:]])

            # get iou
            w = np.maximum(0.0, xx2 - xx1 + 0.01)
            h = np.maximum(0.0, yy2 - yy1 + 0.01)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return np.array(rest)
