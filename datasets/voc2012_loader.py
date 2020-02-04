from torchvision.datasets import VOCDetection, VOCSegmentation
import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random

train_data_dir = '/home/jrlees/datasets/VOC2012/'
train_anno_dir = '/home/jrlees/datasets/VOC2012/VOCdevkit/VOC2012/Annotations/'
train_img_dir = '/home/jrlees/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages/'

datas_train = VOCDetection(root=train_data_dir, image_set='train', download=False,
                           transform=T.Compose([ T.RandomHorizontalFlip(0.5), T.Resize((448, 448)), T.ToTensor()]))
masks_train = VOCSegmentation(root=train_data_dir, image_set='train', download=False,
                              transform=T.Compose([T.Resize((448, 448)), T.ToTensor(),
                                                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                              )
datas_val = VOCDetection(root=train_data_dir, image_set='val', download=False,
                         transform=T.Compose([T.Resize((448, 448)), T.ToTensor()]))
masks_val = VOCSegmentation(root=train_data_dir, image_set='val', download=False,
                            transform=T.Compose([T.Resize((448, 448)), T.ToTensor(),
                                                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
object_index = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus',
                'motorbike', 'car', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


class VOCDataset(Dataset):
    def __init__(self, image_set='train'):
        if image_set == 'train':
            self.data = datas_train
            self.mask = masks_train
        else:
            self.data = datas_val
            self.mask = masks_val

    def __getitem__(self, item):
        images, target = self.data[item]
        name = target['annotation']['filename']
        bboxs = []
        object_class = []
        objects = target['annotation']['object']
        import os
        img = Image.open(os.path.join(train_img_dir, name))
        h, w = img.size
        raitox, ratioy = 448 / h, 448 / w
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            bbox = object['bndbox']
            bbox = [eval(bbox['xmin']) * raitox, eval(bbox['ymin']) * ratioy,
                    eval(bbox['xmax']) * raitox, eval(bbox['ymax']) * ratioy]
            bboxs.append(bbox)
            object_class.append(object_index.index(object['name']))
        # print(name)
        # name = [name]
        # print(type(images))
        # print(type(object_class))
        # print(len(bboxs))
        return images, np.array(bboxs), np.array(object_class), name, np.array([raitox, ratioy], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, bboxes, object_classes, name, ratios = list(zip(*batch))
        images = torch.stack([img for img in images], dim=0)
        names = [na for na in name]
        idxs = []
        bbox = []
        ratioss = []
        for i, batch_box in enumerate(bboxes):
            for _ in batch_box:
                idxs.append(i)
                ratioss.append(ratios[i])
            # print('bbox', batch_box.shape)
            # print('ocls', batch_object_class.shape)
            bbox.append(batch_box)
        # print(object_classes)
        ocl = np.vstack([ocls.reshape(-1, 1) for ocls in object_classes])
        # print(bbox)
        bbox = np.vstack(bbox)
        bbox = np.hstack([bbox, np.array(idxs).reshape(-1, 1)])
        ocls = np.hstack([ocl.reshape(-1, 1), np.array(idxs).reshape(-1, 1)])
        return images, bbox.astype(np.float32), ocls, names, ratioss



