import sys
import os
import copy
import torch
import torch.nn as nn
import time
import cv2

from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import datasets
from detection.detection_model import Detection
import datasets
from detection.config import get_config
from detection.utils import *


class FineTune:
    def __init__(self, config, backbone='fasterrcnn', num_classes=20):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.build_model(backbone=backbone)
        self.set_parameter_requires_grad()
        self.build_optimizer()

    def build_optimizer(self):
        paras_to_update = self.model.parameters()
        print("Parameters to learn:")
        if not self.config.feature_extracting:
            paras_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    paras_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        self.opt = torch.optim.Adam(paras_to_update, lr=self.config.learning_rate,betas=(0.9, 0.999), weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=2, gamma=0.1)

    def set_parameter_requires_grad(self):
        if not self.config.feature_extracting:
            for param in self.model.named_parameters():
                layer_name = param[0]
                # print(layer_name)
                if layer_name.split('.')[1] == 'backbone':
                    param[1].requires_grad = False

    def build_model(self, backbone):
        self.model = Detection(backbone=backbone, use_pretrained=True).to(self.device)
        in_features = self.model.detection_backbone.roi_heads.box_predictor.cls_score.in_features
        self.model.detection_backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes).to(self.device)
        if self.config.restore:
            self.model.load_state_dict(torch.load(os.path.join(self.config.model_path, self.config.model_name +'.pts')))

    def evaluate(self, testloader, iou_thres=0.5, conf_thres=0.001, nms_thres=0.7):
        print('Start evaluating!!')
        self.model.eval()
        labels = []
        sample_metrics = [] # List of tuples (TP, confs, pred)

        for batch_i, (batch_image, batch_bbox, batch_class, batch_name, batch_ratio) in enumerate(tqdm.tqdm(testloader, desc="Evaluation")):
            # Extract labels
            # print(batch_class[:, 0].tolist())
            # print(len(labels))
            labels += batch_class[:, 0].tolist()
            # print(len(labels))
            target_bbox = batch_bbox[:, :4] / np.tile(batch_ratio, (1, 2))

            batch_out_bbox = []
            batch_out_score = []
            batch_out_label = []
            batch_image = batch_image.to(self.device)
            with torch.no_grad():
                outputs = self.model(list(batch_image))
                for i, output in enumerate(outputs):
                    score = output['scores'].cpu().numpy()
                    bboxs = output['boxes'].cpu().numpy()
                    label = output['labels'].cpu().numpy()
                    idx = self.model.nms_postprocess(bboxs=bboxs, scores=score, threshold=nms_thres)
                    batch_out_bbox.append(bboxs[idx] / np.tile(batch_ratio[i], (1, 2)) if idx.shape[0] != 0 else idx)
                    batch_out_score.append(score[idx] if idx.shape[0] != 0 else idx)
                    batch_out_label.append(label[idx] if idx.shape[0] != 0 else idx)

            targets = np.hstack([batch_class[:, ::-1], target_bbox])
            targets = torch.from_numpy(targets)
            sample_metrics += get_batch_statistics(len(outputs), (batch_out_bbox,
                                                                  batch_out_score, batch_out_label), targets, iou_threshold=iou_thres)

        tps, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(tps, pred_scores, pred_labels, labels)
        # logger = 'Evaluation precision = {:.4f} | recall = {:.4f} | mAP = {:.4f} | F1-score = {:.4f} | mAP-class = {:.4f}'.format(
        #     precision, recall, AP, f1, ap_class
        # )
        # print(logger)
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' () - AP: {AP[i]}  F1-score: {f1[i]}")

        print(f"mAP: {AP.mean()}")
        return precision, recall, AP, f1, ap_class

    def train_model(self, dataloaders):
        begin = time.time()
        for epoch in range(self.config.num_epochs):
            print("Epoch {} / {} ".format(epoch, self.config.num_epochs - 1))
            print("-"*10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()

                running_loss = 0.0
                batch_counter = 0
                if phase == 'val':
                    self.evaluate(dataloaders[phase])
                    continue
                for j, (batch_image, batch_bbox, batch_class, batch_name, _) in enumerate(tqdm.tqdm(dataloaders[phase], desc='Training process')):
                    batch_image = batch_image.to(self.device)
                    batch_target = []
                    # print(batch_image.shape[0])
                    for i in range(batch_class[-1, -1] + 1):
                        target = {}
                        target['boxes'] = torch.from_numpy(batch_bbox[batch_bbox[:, 4] == i][:, :4]).to(self.device)
                        target['labels'] = torch.from_numpy(batch_class[batch_class[:, 1] == i][:, 0].flatten()).to(self.device)
                        batch_target.append(target)
                    # print(len(batch_target))
                    if batch_image.shape[0] != len(batch_target):
                        print(batch_target)
                        exit(-1)
                    self.opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            # print(batch_image.shape)
                            try:
                                loss = self.model(X=list(batch_image), target=batch_target)
                            except ValueError:
                                print(batch_name)
                                print(batch_bbox)
                                exit(-1)
                            # print(loss)
                            this_loss = loss['loss_classifier'] + loss['loss_box_reg'] \
                                        + 0.1 * loss['loss_objectness'] + 0.001 * loss['loss_rpn_box_reg']
                            during = time.time() - begin
                            if batch_counter % 20 == 0:
                                print('Time: {:.0f}m {:.0f}s | batch {:d}/{:d} | train loss: {:.4f}'.
                                      format(during // 60, during % 60, batch_counter,
                                             len(dataloaders[phase]), this_loss))

                        # print(loss['loss_classifier'], loss['loss_box_reg'], loss['loss_objectness'], loss['loss_rpn_box_reg'])
                        # print(loss['loss_box_reg'])
                        # print(loss['loss_objectness'])
                        # print(loss['loss_rpn_box_reg'])
                    this_loss.backward()
                    if batch_counter % 5 == 0:
                        import random
                        c = random.choice(list(range(10)))
                        if c >= 7:
                            nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
                    self.opt.step()
                    self.lr_scheduler.step(epoch)

                    running_loss += this_loss.item() * batch_image.size(0)
                    batch_counter += 1

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                # epoch_acc = running_correct / len(dataloaders[phase].dataset)
                print('-'*15)
                if phase == 'train':
                    logger = '{} Loss: {:.4f}'.format(phase, epoch_loss)
                    print(logger)
                print('\n')

        time_elapsed = time.time() - begin
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print('Best acc {:.4f}'.format(best_acc))

        torch.save(self.model.state_dict(), os.path.join(self.config.model_path, self.config.model_name +'.pts'))

    # def draw_blocks(self, loader, mode='target'):
    #     for batch_i, (batch_image, batch_bbox, batch_class, batch_name, batch_ratio) in enumerate(tqdm.tqdm(loader, desc="drawing boxes")):
    #         pass
    #
    # def draw_block(self, bboxs, name, save_dir, ratios):
    #     root_dir = os.path.join(train_ds.data_dir, '../visualization/')
    #     img_name = os.path.join(root_dir, name)
    #     img = cv2.imread(img_name)
    #     for bbox, ratio in zip(bboxs, ratios):
    #         resize_bbox = bbox / np.tile(ratios, (1, 2))
    #         cv2.rectangle(img, (resize_bbox[0], resize_bbox[1]), (resize_bbox[2], resize_bbox[3]), (0,255,0), thickness=1)
    #     if not os.path.exists(root_dir):
    #         os.mkdir(root_dir)
    #     cv2.imwrite(os.path.join(save_dir, name), img)


if __name__ == '__main__':
    config = get_config()

    if config.dataset == 'VOC':
        train_ds = datasets.VOCDataset(image_set='train')
        val_ds = datasets.VOCDataset(image_set='val')
        train_instance = FineTune(config)
    elif config.dataset == 'COCO':
        train_ds = datasets.CocoDataset(image_set='train')
        val_ds = datasets.CocoDataset(image_set='val')
        train_instance = FineTune(config, num_classes=91)
    else:
        train_ds = datasets.ICDAR_MultiL(image_set='train')
        val_ds = datasets.ICDAR_MultiL(image_set='val')
        train_instance = FineTune(config, num_classes=14)

    train_loader = DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2,
                              collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(dataset=val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2,
                            collate_fn=val_ds.collate_fn)
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)
    train_instance.train_model(dataloaders)
