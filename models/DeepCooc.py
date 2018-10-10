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
        self.conv1x1 = nn.Conv2d(2048, 128, kernel_size = 1, bias = False)
        self.cooc = Cooc_Layer(128)
        # Feature map reduction to remove features that have no contribution to information
        # and reduce computational cost
        self.last_linear = nn.Linear(128*8*8, num_classes)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.conv1x1(x))
        # cooc_block
        #g_x, l1_loss = self.cooc(x)
        g_x = self.cooc(x)
        x = self.gamma * g_x + x

        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)

        rets = self.last_linear(x)
        return rets

if __name__ == '__main__':
    print(" [*] Deep cooc model forward test")
    x = torch.randn([8, 3, 448, 448])
    net = DeepCooc(200)
    logits, l1_loss = net(x)
    print(" [*] output shape:", logits.size())
