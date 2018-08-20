import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Cooc_Layer(nn.Module):
    def __init__(self, in_channels):
        """
            Co-occurrence block that contains conv1x1

            Args:
                in_channels: number of input channels, scalar(integer)

            Return:
                x: Tensor, [batch, in_channels, h, w] shape, that after the squence of operations.
        """
        super(Cooc_Layer, self).__init__()

        self.depth = in_channels
        self.conv1x1 = nn.Conv2d(in_channels-1, 1, kernel_size = 1, bias = False)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, w, h = x.size()
        splited = x.split(1, 1) # in_channels number of [batch, 1, w, h]
        g_features = []
        losses = []

        contexts = []
        for i in range(self.depth):
            context = None
            if i == 0:
                context = torch.cat(splited[1:self.depth], 1)
            elif i == (self.depth-1):
                context = torch.cat(splited[0:self.depth-1], 1)
            else:
                p1 = torch.cat(splited[0:i], 1)
                p2 = torch.cat(splited[i+1:self.depth], 1)
                context = torch.cat([p1, p2], 1)
            contexts.append(context)
        contexts = torch.stack(contexts, 1)
        contexts = contexts.view(-1, self.depth-1, w, h)

        g_features = self.conv1x1(contexts)
        g_features = g_features.view(b, c, w, h)
        #losses = F.smooth_l1_loss(g_features, x.detach())
        losses = torch.sqrt(torch.sum(((g_features - x.detach())**2))) * 0.0025

        """
        for i in range(self.depth):
            context = None
            m = splited[i].contiguous()
            if i == 0:
                context = torch.cat(splited[1:self.depth], 1)
            elif i == (self.depth-1):
                context = torch.cat(splited[0:self.depth-1], 1)
            else:
                p1 = torch.cat(splited[0:i], 1)
                p2 = torch.cat(splited[i+1:self.depth], 1)
                context = torch.cat([p1, p2], 1)
            g_feature = self.relu(self.conv1x1(context))
            g_features.append(g_feature)
            loss = F.smooth_l1_loss(g_feature, m.detach())
            losses.append(loss)

        g_features = torch.cat(g_features, 1)
        losses = torch.stack(losses, 0).sum()
        """

        return g_features, losses

if __name__ == '__main__':
    print(" [*] Cooc layer forward test")
    x = torch.randn([8, 128, 11, 11])
    cooc_layer = Cooc_Layer(128)
    ret, loss = cooc_layer(x)

    print(" [*] output shape:", ret.size())
    print(" [*] loss shape:", loss.size(), "loss value:", loss)
