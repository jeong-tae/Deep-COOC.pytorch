import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        self.relu = nn.ReLU()
        self.eps = 1e-11
        # g_filter is const. we can reduce memory a little bit more if we move out this code as global
        # but it may harm to generalization of code.
        self.g_filter = nn.Conv2d(out_channels, out_channels, kernel_size = 3, groups = out_channels, bias = False)
        x_cord = torch.arange(3)
        x_grid = x_cord.repeat(3).view(3, 3)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim = -1)

        mean = (3 - 1) / 2.
        variance = 0.5**2
        gaussian_kernel = (1./(2. * math.pi * variance)) *\
                                        torch.exp(
                                            -torch.sum((xy_grid - mean) **2., dim = -1) /\
                                            (2 * variance)
                                        )
        gaussian_kernel = gaussian_kernel.view(1, 1, 3, 3).repeat(out_channels, 1, 1, 1)
        self.g_filter.weight.data = gaussian_kernel
        self.g_filter.weight.requires_grad = False

    def forward(self, x):
        
        x = self.conv1x1(x)
        x = self.relu(x)
        
        #TODO: padding

        # 1. why do we need to do padding?
        
        # gaussian filitering
        x = self.g_filter(x)

        # Correlation filtering
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        xshift = []
        for i in range(h*w): # num_shift
            shifted = self.roll(x, i, 2)
            xshift.append(shifted)
        xshift = torch.cat(xshift, 1).view(b, h*w, c*h*w)
        # (b, num_shift, c*c)
        responses = torch.bmm(x, xshift).view(b, h*w, c*c)
        c_ij, o_tmp = torch.max(responses, 1)
        o_tmp = o_tmp.view(b, c, c)
        o_x, o_y = o_tmp / w, o_tmp % h
        o_ij = torch.stack([o_x, o_y], -1)
        
        # square root
        c_ij = torch.sqrt(c_ij)
        # L2 normalize
        norm = c_ij.pow(2).sum(dim = 1, keepdim = True) + self.eps
        c_ij = c_ij / norm

        return c_ij

    def roll(self, tensor, shift, axis):
    
        if shift == 0:
            return tensor

        if axis < 0:
            axis += tensor.dim()

        dim_size = tensor.size(axis)
        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)

        before = tensor.narrow(axis, 0, dim_size - shift)
        after = tensor.narrow(axis, after_start, shift)
        return torch.cat([after, before], axis)

if __name__ == '__main__':
    print(" [*] Cooc layer forward test")
    x = torch.randn([8, 2048, 11, 11])
    cooc_layer = Cooc_Layer(2048, 128)
    ret = cooc_layer(x)

    print(" [*] output shape:", ret.size())
