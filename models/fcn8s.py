import os.path as osp

import fcn
import torch.nn as nn

from .fcn32s import get_upsampling_weight


class FCN8s(nn.Module):

    def __init__(self, n_class=21):
        super().__init__()
        self.model_name = 'FCN8s'
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool2 = nn.Conv2d(128, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        self.upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 16, stride=4)
        self.upscore3 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2)
        self.upscore4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2) # c = 256, 1/8
        conv4 = self.conv_block4(conv3) # c = 512, 1/16
        score4 = self.classifier(conv4)

        upscore4 = self.upscore4(score4)
        score3 = self.score_pool3(conv3)
        score3 = score3[:, :, 5:5 + upscore4.size()[2], 5:5 + upscore4.size()[3]].contiguous()
        score3 += upscore4

        upscore3 = self.upscore3(score4)
        score2 = self.score_pool2(conv2)
        score2 = score2[:, :, 9:9 + upscore3.size()[2], 9:9 + upscore3.size()[3]].contiguous()
        score2 += upscore3

        out = self.upscore(score2)
        out = out[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        return out