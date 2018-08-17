import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .ResNet import resnet50, resnet152
from .cooc_layer import Cooc_Layer

class DeepCooc(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(DeepCooc, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False # make sure

        self.backbone = resnet152(num_classes = 1000, pretrained = pretrained)
        self.cooc_a = Cooc_Layer(2048, 128)
        self.cooc_b = Cooc_Layer(2048, 128)
        self.cooc_c = Cooc_Layer(2048, 128)
        # Feature map reduction to remove features that have no contribution to information
        # and reduce computational cost
        self.last_linear = nn.Linear((2048*8*8) + (128*128*3), num_classes)

    def forward(self, x):
        x, ax, bx, cx = self.backbone(x)

        # cooc_block
        ax = self.cooc_a(ax)
        bx = self.cooc_b(bx)
        cx = self.cooc_c(cx)

        concat = torch.cat([x, ax, bx, cx], 1)
        rets = self.last_linear(concat)
        return rets

if __name__ == '__main__':
    print(" [*] Deep cooc model forward test")
    x = torch.randn([8, 3, 448, 448])
    net = DeepCooc(200)
    logits = net(x)
    print(" [*] output shape:", logits.size())
