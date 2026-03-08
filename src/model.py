
import os
# from cda_comp.src.baseline_simple_cnn import train_and_evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import transforms, models
import timm  # For ViT
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Data transforms
from torchvision.transforms import GaussianBlur, ColorJitter, RandomErasing
from sklearn.metrics import confusion_matrix
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import RandAugment, RandomAffine
# from ultralytics import YOLO
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # After two pools: 224/4 = 56
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class FeedBunkClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.base = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)  # ViT for resilience
        self.fc = nn.Sequential(
            nn.Linear(self.base.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.base(x)
        return self.fc(features)
    


# class ResNet50Classification(nn.Module):
#     """
#     ResNet-50 with ImageNet pretrained weights + custom head
#     Moderate-large size, very stable training
#     """
#     def __init__(self, num_classes=5, dropout=0.4):
#         super().__init__()
#         self.base = models.resnet50(weights='IMAGENET1K_V2')   # or IMAGENET1K_V1

#         # Replace final FC layer
#         in_features = self.base.fc.in_features  # 2048
#         self.base.fc = nn.Identity()

#         self.head = nn.Sequential(
#             nn.Linear(in_features, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         features = self.base(x)
#         return self.head(features)

class ResNet50Classification(nn.Module):
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        self.base = models.resnet50()

        # Freeze everything first
        for param in self.base.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks (layer3 and layer4)
        for param in self.base.layer3.parameters():
            param.requires_grad = True
        for param in self.base.layer4.parameters():
            param.requires_grad = True

        in_features = self.base.fc.in_features  # 2048
        self.base.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),          # ← added BN
            nn.ReLU(),
            nn.Dropout(dropout),          # increased to 0.5
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.base(x)
        return self.head(features)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-Excitation block – adds attention-like complexity with very few parameters"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class BottleneckDenseBlock(nn.Module):
    """Dense block with bottleneck for parameter efficiency"""
    def __init__(self, in_channels, growth_rate, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, bias=False),  # 1x1 bottleneck
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False),
            SEBlock(growth_rate),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.block(x)
        return torch.cat([x, out], dim=1)  # dense connection


class MiniDenseNet(nn.Module):
    """
    Compact DenseNet-style model for small datasets – trains from scratch
    ~1.8M parameters (with growth_rate=24)
    Hard-coded channel sizes to avoid attribute errors
    """
    def __init__(self, num_classes=5, growth_rate=24, dropout=0.4):
        super().__init__()

        # ──────────────────────────────────────────────
        # Entry block
        # ──────────────────────────────────────────────
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ──────────────────────────────────────────────
        # Dense Block 1
        # Input channels: 64
        # Output after 8 layers: 64 + 8×24 = 256
        # ──────────────────────────────────────────────
        self.dense1 = self._make_dense_block(64, growth_rate, num_layers=8, dropout=dropout)

        # Transition 1: compress 256 → 128
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ──────────────────────────────────────────────
        # Dense Block 2
        # Input channels: 128
        # Output after 8 layers: 128 + 8×24 = 320
        # ──────────────────────────────────────────────
        self.dense2 = self._make_dense_block(128, growth_rate, num_layers=8, dropout=dropout)

        # Transition 2: compress 320 → 160
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
            nn.Conv2d(320, 160, kernel_size=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ──────────────────────────────────────────────
        # Dense Block 3 (final dense block)
        # Input channels: 160
        # Output after 8 layers: 160 + 8×24 = 352
        # ──────────────────────────────────────────────
        self.dense3 = self._make_dense_block(160, growth_rate, num_layers=8, dropout=dropout)

        # ──────────────────────────────────────────────
        # Global classifier
        # ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(352, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout + 0.1),
            nn.Linear(256, num_classes)
        )

        # Initialize weights (important for training from scratch)
        self._initialize_weights()

    def _make_dense_block(self, in_channels, growth_rate, num_layers, dropout):
        layers = []
        current_channels = in_channels
        for _ in range(num_layers):
            layers.append(BottleneckDenseBlock(current_channels, growth_rate, dropout))
            current_channels += growth_rate
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.entry(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.classifier(x)
        return x