import numpy as np
import cv2
import random


class DrawText:
    def __init__(self, root_dir, languages=2):
        self.root_dir = root_dir
        self.texts = []
        self.sizes = [8, 10, 12, 13, 14, 16, 20, 32]

    def add_text(self, image):
        img = cv2.imread(image)
        h, w = img.shape[:2]
        cv2.putText(img, )

    def update_pos(self):
        pass

    def update_label(self):
        pass

