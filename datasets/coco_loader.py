from torchvision.datasets import CocoDetection
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from detection.utils import xywh2xyxy
import os

train_data_dir = '/home/jrlees/datasets/COCO/images/train2017/'
train_anno_dir = '/home/jrlees/datasets/COCO/annotations/instances_train2017.json'
val_data_dir = '/home/jrlees/datasets/COCO/images/val2017/'
val_anno_dir = '/home/jrlees/datasets/COCO/annotations/instances_val2017.json'

data_train = CocoDetection(root=train_data_dir, annFile=train_anno_dir,
                           transform=T.Compose([T.RandomHorizontalFlip(0.5), T.Resize((640, 640)), T.ToTensor()]))
data_val = CocoDetection(root=val_data_dir, annFile=val_anno_dir,
                         transform=T.Compose([T.Resize((640, 640)), T.ToTensor()]))

class CocoDataset(Dataset):
    def __init__(self, image_set='train'):
        # 这个函数一般是先初始化一些如位置信息以及resize的大小等
        if image_set == 'train':
            self.data = data_train
            self.data_dir = train_data_dir
        else:
            self.data = data_val
            self.data_dir = val_data_dir

    def __getitem__(self, item):
        # 一般是按顺序读取图片
        images, target = self.data[item]
        # Only consider detection
        bboxs = []
        object_class = []
        if len(target) == 0:
            return images, np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.long), [], np.zeros((0, 2), dtype=np.float32)
        name = '{:012d}.jpg'.format(target[0]['image_id'])

        img = Image.open(os.path.join(self.data_dir, name))
        h, w = img.size
        raitox, ratioy = 640 / h, 640 / w
        for dic in target:
            bbox = dic['bbox']
            bboxs.append([bbox[0]*raitox, bbox[1] * ratioy, bbox[2]*raitox, bbox[3]*ratioy])
            object_class.append(dic['category_id'])
        # print(np.array(bboxs))
        # print(xywh2xyxy(np.array(bboxs)))

        return images, xywh2xyxy(np.array(bboxs)), np.array(object_class), name, np.array([raitox, ratioy], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, bboxes, object_classes, name, ratios = list(zip(*batch))
        valid_images = []
        # images = torch.stack([img for img in images], dim=0)
        valid_names = []
        idxs = []
        bbox = []
        ratioss = []
        counter = 0
        batch_counter = 0
        for (batch_box, img, na) in zip(bboxes, images, name):
            if batch_box.shape[0] > 0:
                valid_images.append(img)
                valid_names.append(na)
            else:
                counter += 1
                continue
            for _ in batch_box:
                idxs.append(batch_counter)
                ratioss.append(ratios[counter].reshape(-1, 2))
            # print('bbox', batch_box.shape)
            # print('ocls', batch_object_class.shape)

            bbox.append(batch_box)
            counter += 1
            batch_counter += 1

        # print(object_classes)
        ocl = np.vstack([ocls.reshape(-1, 1) for ocls in object_classes])
        # print([k.shape for k in ratioss])
        # print(ratioss)
        ratioss = np.vstack(ratioss)
        # print(ratioss.shape)
        # exit(-1)
        # print(bbox)
        bbox = np.vstack(bbox)
        bbox = np.hstack([bbox, np.array(idxs).reshape(-1, 1)])
        ocls = np.hstack([ocl.reshape(-1, 1), np.array(idxs).reshape(-1, 1)])
        images = torch.stack(valid_images, 0) if len(valid_images) != 0 else torch.stack([img for img in images], 0)
        return images, bbox.astype(np.float32), ocls, valid_names, ratioss
