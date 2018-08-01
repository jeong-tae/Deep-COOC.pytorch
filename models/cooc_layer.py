import torch
import torch.nn as nn
import torch.nn.functional as F


class Cooc_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
            Co-occurrence block that contains conv1x1, sqrt and L2norm
            L2 norm is applied to each channels

            Args:
                in_channels: number of input channels, scalar(integer)
                out_channels: number of output channels, scalar(integer)

            Return:
                x: Tensor, [batch, ch, h, w] shape, that after the squence of operations.
        """
        super(Cooc_Layer, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False)
        self.eps = 1e-11


        # g_filter is const. we can reduce memory a little bit more if we move out this code as global
        # but it may harm to generalization of code.
        self.g_filter = nn.Conv2d(out_channels, out_chnnels, kernel_size = 3, groups = out_channels, bias = False)
        x_cord = torch.arange(3)
        x_grid = x_cord.repeat(3).view(3, 3)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim = -1)

        mean = (3 - 1) / 2.
        variance = 0.5**2

        self.g_filter.weight.data = (1./(2. * math.pi * variance))
                                        * torch.exp(-torch.sum((xy_grid - mean) ** 2.,
                                        dim = -1) / (2 * variance)
        self.g_filter.weight.requires_grad = False

    def forward(self, x):
        
        x = self.conv1x1(x)
        
        #TODO: padding, gaussian filtering, calc c_ija

        # 1. why do we need to do padding?


        x = self.g_filter(x)

        # co-occurrence correlation
        # to do this, shift the feature n^2 times and stack them to compute once at all
        # get the maximum value of output on channels?


        x = torch.sqrt(x)
        norm = x.pow(2).sum(dim = 1, keepdim = True) + self.eps
        x = x / norm

        return x
