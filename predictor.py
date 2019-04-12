# -^- coding:utf-8 -^- 

"""
The predictor is used to predict the loss gained by doing downsampling
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.utils.model_zoo as model_zoo

class Predictor(nn.Module):
    def __init__(self,inplanes,hidden=16):
        super(Predictor, self).__init__()
        # the network structure is a 1x1 conv layer followed by a 3x3 conv layer, after global pooling, put into a FC
        self.conv1 = nn.Conv2d(inplanes, hidden, kernel_size=1, bias=True)
		# self.bn1 = nn.BatchNorm2d(D*C)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, stride=1,
							   padding=1, bias=True)
        self.fc = nn.Linear(hidden, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    