#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
from spectral import SpectralNorm

#Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 8, 5, 1, 2)),
            nn.BatchNorm2d(8),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(8, 16, 5, 1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(16, 64, 5, 1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, 5, 1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 128, 5, 1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 128, 5, 1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.conv_mask = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 1, 5, 1, 2))
            )
        self.conv7 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 64, 5, 4, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 32, 5, 4, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.average_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            SpectralNorm(nn.Linear(32, 1)),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        # x = self.conv7(x)
        # print 7, x.shape
        x = self.conv8(x)
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
