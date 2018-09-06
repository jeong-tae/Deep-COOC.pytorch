import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .ResNet import resnet50, resnet101, resnet152

class Resnet152(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(Resnet152, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False

        self.net = resnet152(num_classes = 1000, pretrained = pretrained)
        self.last_linear = nn.Linear(2048 * 8 * 8, num_classes)

    def forward(self, x):
        x, _, _, _ = self.net(x)
        rets = self.last_linear(x)
        return rets

class Resnet50(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(Resnet50, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False

        self.net = resnet50(num_classes = 1000, pretrained = pretrained)
        self.last_linear = nn.Linear(2048 * 8 * 8, num_classes)

    def forward(self, x):
        x, _, _, _ = self.net(x)
        rets = self.last_linear(x)
        return rets
