from torchvision.datasets import CocoDetection
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os

train_root_dir = '/home/jrlees/datasets/COCO/images/textOCR/ImagesPart1/'
anno_dir = '/home/jrlees/datasets/COCO/images/textOCR/train_gt_t13/'
val_root_dir = '/home/jrlees/datasets/COCO/images/textOCR/ImagesPart2/'
LANGUAGE_INDEX = ['Chinese', 'Japanese', 'Latin', 'Symbols', 'English',
                  'Arabic', 'German', 'French', 'Korean', 'None', 'Mixed'
                  'Italian', 'Bangla', 'Hindi']


class ICDAR_MultiL(Dataset):
    def __init__(self, image_set='train'):
        if image_set == 'train':
            self.data_dirs = list(os.listdir(train_root_dir))
            self.transform = T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.Resize((448, 448)),
                T.ToTensor()
            ])
        else:
            self.data_dirs = list(os.listdir(val_root_dir))
            self.transform = T.Compose([
                T.Resize((448, 448)),
                T.ToTensor()
            ])
        self.anno_dir = anno_dir

    def __getitem__(self, item):
        # 不加mask的版本
        img_name = self.data_dirs[item]
        image_path = os.path.join(train_root_dir, img_name)
        anno_path = os.path.join(anno_dir, img_name[:-4] + '.txt')
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        ratiox, ratioy = 448 / w, 448 / h
        boxs, classes = [], []
        lines = open(anno_path, 'r', encoding='UTF-8').read().splitlines()
        for li in lines:
            valid = li.split(',')[:9]
            lang = valid[8]
            try:
                classes.append(LANGUAGE_INDEX.index(lang))
            except ValueError:
                print(li)
                exit(-1)
            boxes = list(eval(','.join(valid[:-1])))
            xmin = np.floor(min(boxes[0], boxes[6]))
            ymin = np.floor(min(boxes[1], boxes[3]))
            xmax = np.ceil(min(boxes[2], boxes[4]))
            ymax = np.ceil(min(boxes[5], boxes[7]))
            boxs.append([xmin, ymin, xmax, ymax])
        return self.transform(img), np.array(boxs, dtype=np.float32), np.array(classes, dtype=np.long), img_name, np.array([ratiox, ratioy], dtype=np.float32)

    def __len__(self):
        return len(self.data_dirs)

    def collate_fn(self, batch):
        images, bboxes, object_classes, name, ratios = list(zip(*batch))
        images = torch.stack([img for img in images], dim=0)
        names = [na for na in name]
        idxs = []
        bbox = []
        ratioss = []
        for i, batch_box in enumerate(bboxes):
            print(ratios[i])
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

