import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.hfgnet import HFGNet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class HFGNetSegmentation(object):
    _defaults = {
        "model_path": "",
        "num_classes": 2,
        "backbone": "HFGNet_W18_Small",
        "input_shape": [256, 256],
        "cuda": True,
    }
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes == 2:
            self.colors = [(0, 0, 0), (255, 255, 255)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):
        self.net = HFGNet(num_classes=self.num_classes, backbone=self.backbone)

        self.cuda = self.cuda and torch.cuda.is_available()
        if self.model_path:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        nw = self.input_shape[1]
        nh = self.input_shape[0]
        image_data = image.resize((nw, nh), Image.BICUBIC)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            mask = np.uint8(seg_img)
            return mask

    def detect_image_pro(self, image):
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]
        nw = self.input_shape[1]
        nh = self.input_shape[0]
        image_data = image.resize((nw, nh), Image.BICUBIC)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr_resized = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            probability_maps = []
            for i in range(self.num_classes):
                prob_map = pr_resized[:, :, i]  # 提取第 i 类的概率图
                probability_maps.append(prob_map)

            pr = pr_resized.argmax(axis=-1)
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                                 [original_h, original_w, -1])
            mask = np.uint8(seg_img)
            return probability_maps




    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

