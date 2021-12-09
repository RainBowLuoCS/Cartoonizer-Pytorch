# -*- coding=utf-8 -*-
# @Time :2021/12/8 12:54
# @Author :Hobbey
# @Site : 
# @File : Cartoonize.py
# @Software : PyCharm

import numpy as np
import torch
import cv2
import os

import torch.utils.data
from torch.nn import functional as F

import math
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


class ResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, name="res_block"):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.name = name
        self.c1 = Conv2d(self.in_channel, self.out_channel, [3, 3], stride=1)
        self.r1 = torch.nn.LeakyReLU(0.2)
        self.c2 = Conv2d(self.in_channel, self.out_channel, [3, 3], stride=1)

    def forward(self, x):
        out = self.c1(x)
        out = self.r1(out)
        out = self.c2(out)
        return out + x


class UNetCartoon(torch.nn.Module):
    def __init__(self, in_channel, out_channel=32, num_blocks=4, name='generator'):
        super(UNetCartoon, self).__init__()
        self.inchannel = in_channel
        self.outchannel = out_channel
        self.num_blocks = num_blocks
        self.name = name
        self.c1 = Conv2d(self.inchannel, self.outchannel, [7, 7])
        self.r1 = torch.nn.LeakyReLU(0.2)
        self.c2 = Conv2d(self.outchannel, self.outchannel, [3, 3], stride=2)
        self.r2 = torch.nn.LeakyReLU(0.2)
        self.c3 = Conv2d(self.outchannel, 2 * self.outchannel, [3, 3])
        self.r3 = torch.nn.LeakyReLU(0.2)
        self.c4 = Conv2d(2 * self.outchannel, 2 * self.outchannel, [3, 3], stride=2)
        self.r4 = torch.nn.LeakyReLU(0.2)
        self.c5 = Conv2d(2 * self.outchannel, 4 * self.outchannel, [3, 3])
        self.r5 = torch.nn.LeakyReLU(0.2)
        self.res1 = ResBlock(4 * self.outchannel, 4 * self.outchannel, name='block_{}'.format(1))
        self.res2 = ResBlock(4 * self.outchannel, 4 * self.outchannel, name='block_{}'.format(2))
        self.res3 = ResBlock(4 * self.outchannel, 4 * self.outchannel, name='block_{}'.format(3))
        self.res4 = ResBlock(4 * self.outchannel, 4 * self.outchannel, name='block_{}'.format(4))
        self.c6 = Conv2d(4 * self.outchannel, 2 * self.outchannel, [3, 3])
        self.r6 = torch.nn.LeakyReLU(0.2)
        self.u6 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.c7 = Conv2d(2 * self.outchannel, 2 * self.outchannel, [3, 3])
        self.r7 = torch.nn.LeakyReLU(0.2)
        self.c8 = Conv2d(2 * self.outchannel, self.outchannel, [3, 3])
        self.r8 = torch.nn.LeakyReLU(0.2)
        self.u8 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.c9 = Conv2d(self.outchannel, self.outchannel, [3, 3])
        self.r9 = torch.nn.LeakyReLU(0.2)
        self.c10 = Conv2d(self.outchannel, self.inchannel, [7, 7])

    def forward(self, x):
        x0 = self.c1(x)
        x0 = self.r1(x0)

        x1 = self.c2(x0)
        x1 = self.r2(x1)
        x1 = self.c3(x1)
        x1 = self.r3(x1)

        x2 = self.c4(x1)
        x2 = self.r4(x2)
        x2 = self.c5(x2)
        x2 = self.r5(x2)


        x2 = self.res1(x2)
        x2 = self.res2(x2)
        x2 = self.res3(x2)
        x2 = self.res4(x2)



        x2 = self.c6(x2)
        x2 = self.r6(x2)
        x3 = self.u6(x2)

        x3 = self.c7(x1 + x3)
        x3 = self.r7(x3)
        x3 = self.c8(x3)
        x3 = self.r8(x3)
        x4 = self.u8(x3)

        x4 = self.c9(x0 + x4)

        x4 = self.r9(x4)
        x4 = self.c10(x4)
        return x4


class GuideFilter(torch.nn.Module):
    def __init__(self, r=1, eps=1e-2):
        super(GuideFilter, self).__init__()
        self.r = r
        self.eps = eps
        weight = 1 / ((2 * self.r + 1) ** 2)
        self.c1 = Conv2d(3, 3, [2 * self.r + 1, 2 * self.r + 1], bias=False, groups=3)
        torch.nn.init.constant_(self.c1.weight, weight)
        self.c2 = Conv2d(1, 1, [2 * self.r + 1, 2 * self.r + 1], bias=False, groups=1)
        torch.nn.init.constant_(self.c2.weight, weight)

    def forward(self, x, y):
        N = self.c2(torch.ones(1, 1, x.shape[2], x.shape[3]))
        # print(N.shape)
        mean_x = self.c1(x) / N
        mean_y = self.c1(y) / N
        cov_xy = self.c1(x * y) / N - mean_x * mean_y
        var_x = self.c1(x * x) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = self.c1(A) / N
        mean_b = self.c1(b) / N

        output = mean_A * x + mean_b

        return output


class Cartoon(torch.nn.Module):
    def __init__(self):
        super(Cartoon, self).__init__()
        self.unet = UNetCartoon(in_channel=3)
        self.guide = GuideFilter()

    def forward(self, image):
        image = self.__resize_crop(image)
        batch_image = image.astype(np.float32) / 127.5 - 1
        batch_image = np.transpose(batch_image, (2, 0, 1))
        batch_image = np.expand_dims(batch_image, axis=0)
        input = torch.tensor(batch_image,dtype=torch.float32)
        network_output = self.unet(input)
        output = self.guide(input, network_output)
        output = output.detach().numpy()
        output = (np.squeeze(output) + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

    def __resize_crop(self, image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720 * h / w), 720
            else:
                h, w = 720, int(720 * w / h)
        image = cv2.resize(image, (w, h),
                           interpolation=cv2.INTER_AREA)
        h, w = (h // 8) * 8, (w // 8) * 8
        image = image[:h, :w, :]
        return image


if __name__ == '__main__':

    model = Cartoon()
    model.load_state_dict(torch.load('Param/data_param'))
    model.eval()
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    name_list = os.listdir(load_folder)

    for name in (name_list):
        load_path = os.path.join(load_folder, name)
        save_path = os.path.join(save_folder, name)
        image = cv2.imread(load_path)
        output = model(image)
        output = np.transpose(output, (2, 0, 1))
        output = np.transpose(output, (2, 0, 1))
        cv2.imwrite(save_path, output)
