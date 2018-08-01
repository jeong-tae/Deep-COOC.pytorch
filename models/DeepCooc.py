import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .ResNet import resnet50
from cooc_layer import Cooc_layer

class DeepCooc(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(DeepCooc, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False # make sure

        self.backbone = resnet50(num_classes = 1000, pretrained = pretrained)
        self.cooc_a = Cooc_Layer(2048, 128)
        self.cooc_b = Cooc_Layer(2048, 128)
        self.cooc_c = Cooc_Layer(2048, 128)
        # Feature map reduction to remove features that have no contribution to information
        # and reduce computational cost

    def forward(self, x):
        x, ax, bx, cx = self.backbone(x)

        ax = self.cooc_a(ax)
        bx = self.cooc_b(bx)
        cx = self.cooc_c(cx)
        
