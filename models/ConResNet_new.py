import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import scipy.ndimage as ndimage


# class ConvStd(nn.ConvStd):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
#                  groups=8, bias=False):
#         super(ConvStd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
#
#     def forward(self, input: Tensor):
#         weight = self.weight

class ConvStd(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 groups=1, bias=False):
        super(ConvStd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)

        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_conv(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
             bias=False, weight_std=False):
    if weight_std:
        return ConvStd(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                       dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


class AttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), bias=False, weight_std=False, first_layer=False):
        # AttBlock(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        super(AttBlock, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layer = first_layer
        self.weight_std = weight_std

        self.prelu = nn.PReLU()
        self.gn_seg = nn.GroupNorm(8, in_channels)
        self.conv_seg = get_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, bias=bias, weight_std=self.weight_std)
        self.gn_res = nn.GroupNorm(8, out_channels)
        self.conv_res = get_conv(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                 stride=(1, 1, 1), padding=(0, 0, 0), dilation=dilation, bias=bias, weight_std=self.weight_std)

        self.gn_res1 = nn.GroupNorm(8, out_channels)
        self.conv_res1 = get_conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=(1, 1, 1), padding=padding, dilation=dilation, bias=bias, weight_std=self.weight_std)

        self.gn_res2 = nn.GroupNorm(8, out_channels)
        self.conv_res2 = get_conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=(1, 1, 1), padding=padding, dilation=dilation, bias=bias, weight_std=self.weight_std)

        self.gn_mp = nn.GroupNorm(8, in_channels)
        self.conv_mp_first = get_conv(in_channels=4, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, bias=bias, weight_std=self.weight_std)
        self.conv_mp = get_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, bias=bias, weight_std=self.weight_std)

    def _res(self, x):
        # Batch size, channel, D, H, W
        bs, channel, depth, height, width = x.shape
        x_copy = torch.zeros_like(x).cuda()
        x_copy[:, :, 1:, :, :] = x[:, :, 0: depth - 1, :, :]
        res = x - x_copy
        res[:, :, 0, :, :] = 0
        res = torch.abs(res)
        return res

    def forward(self, input):
        x1, x2 = input
        if self.first_layer:
            x1 = self.gn_seg(x1)
            x1 = self.prelu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)

            x2 = self.conv_mp_first(x2)
            x2 = x2 + res

        else:
            x1 = self.gn_seg(x1)
            x1 = self.prelu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)

            if self.in_planes != self.out_planes:
                x2 = self.gn_mp(x2)
                x2 = self.prelu(x2)
                x2 = self.conv_mp(x2)

            x2 = x2 + res

        x2 = self.gn_res1(x2)
        x2 = self.prelu(x2)
        x2 = self.conv_res1(x2)

        x1 = x1 * (1 + torch.sigmoid(x2))

        return [x1, x2]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=(1, 1, 1),
                 downsample=None, fist_dilation=1, multi_grid=1, weight_std=True, group_size=8):
        super(ConvBlock, self).__init__()
        self.weight_std = weight_std
        print(f'conv block weight std = {weight_std}')
        self.downsample = downsample
        self.kernel_size = kernel_size
        self.multi_grid = multi_grid
        # self.stride = stride
        self.dilation = dilation
        # print(f'dilation = {dilation}, type dilation = {type(dilation)}, multi grid = {type(multi_grid)}')
        self.prelu = nn.PReLU()
        self.gn1 = nn.GroupNorm(group_size, in_channels)
        print(f'padding = {dilation * multi_grid}, self.weight_std = {self.weight_std}')
        self.conv1 = get_conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.gn2 = nn.GroupNorm(group_size, out_channels)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

    def forward(self, x):
        skip = x
        print(f'x = {x.size()}')
        seg = self.gn1(x)
        print(f'1 seg = {seg.size()}')
        seg = self.prelu(seg)
        print(f'2 seg = {seg.size()}')
        print(f'padding = {self.dilation * self.multi_grid}')
        seg = self.conv1(seg)
        print(f'3 seg = {seg.size()}')
        seg = self.gn2(seg)
        seg = self.prelu(seg)
        seg = self.conv2(seg)
        print(f'4 seg = {seg.size()}')
        if self.downsample is not None:
            skip = self.downsample(x)

        print(f'ConvBlock: seg = {seg.size()}, skip = {skip.size()}')
        seg = seg + skip

        return seg


class CoConNet(nn.Module):
    def __init__(self, input_size, block, layers, num_filters=32, in_channels=1, num_classes=2, weight_std=False):
        super(CoConNet, self).__init__()
        self.input_size = input_size
        self.weight_std = weight_std
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.block = block
        self.layers = layers
        # print(f'weight std = {weight_std}')
        self.encoder0 = nn.Sequential(
            # Input channel -> num_filters
            # 2 -> 32
            nn.Conv3d(in_channels=in_channels, out_channels=self.num_filters, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False)
        )
        self.encoder00 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=self.num_filters, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False)
        )
        self.encoder1 = nn.Sequential(
            # 32 -> 64
            nn.GroupNorm(8, self.num_filters),
            nn.Conv3d(in_channels=self.num_filters, out_channels=self.num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 2)
        )
        self.encoder11 = nn.Sequential(
            nn.GroupNorm(8, self.num_filters),
            nn.Conv3d(in_channels=self.num_filters, out_channels=self.num_filters * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 2)
        )
        self.encoder2 = nn.Sequential(
            # 64 -> 128
            nn.GroupNorm(8, self.num_filters * 2),
            nn.Conv3d(in_channels=self.num_filters * 2, out_channels=self.num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 4)
        )
        self.encoder22 = nn.Sequential(
            nn.GroupNorm(8, self.num_filters * 2),
            nn.Conv3d(in_channels=self.num_filters * 2, out_channels=self.num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 4)
        )
        self.encoder3 = nn.Sequential(
            # 128 -> 256
            nn.GroupNorm(8, self.num_filters * 4),
            nn.Conv3d(in_channels=self.num_filters * 4, out_channels=self.num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 8)
        )
        self.encoder33 = nn.Sequential(
            nn.GroupNorm(8, self.num_filters * 2),
            nn.Conv3d(in_channels=self.num_filters * 4, out_channels=self.num_filters * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 8)
        )

        # layers = [1, 2, 2, 2, 2]
        self.layer0 = self._make_layer(in_channels=32, out_channels=32, block_num=layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, block_num=layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(in_channels=128, out_channels=128, block_num=layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(in_channels=256, out_channels=256, block_num=layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(in_channels=256, out_channels=256, block_num=layers[4],
                                       stride=(1, 1, 1), dilation=(2, 2, 2))
        # self.layer0 = self._make_layer(in_channels=32, out_channels=32, block_num=layers[0], block=ConvBlock, stride=(1, 1, 1))
        # self.layer1 = self._make_layer(in_channels=64, out_channels=64, block_num=layers[1], block=ConvBlock,stride=(1, 1, 1))
        # self.layer2 = self._make_layer(in_channels=128, out_channels=128, block_num=layers[2], block=ConvBlock, stride=(1, 1, 1))
        # self.layer3 = self._make_layer(in_channels=256, out_channels=256, block_num=layers[3], block=ConvBlock, stride=(1, 1, 1))
        # self.layer4 = self._make_layer(in_channels=256, out_channels=256, block_num=layers[4], block=ConvBlock,
        #                                stride=(1, 1, 1), dilation=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, self.num_filters * 4),
            nn.PReLU(),
            nn.Dropout3d(0.1),
            get_conv(in_channels=self.num_filters * 4, out_channels=self.num_filters * 2, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1))
        )

        self.seg_x4 = nn.Sequential(
            AttBlock(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        self.seg_x2 = nn.Sequential(
            AttBlock(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))
        self.seg_x1 = nn.Sequential(
            AttBlock(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))

        self.seg_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=(1, 1, 1))
        )
        self.res_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=(1, 1, 1))
        )
        self.resx2_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=(1, 1, 1))
        )
        self.resx4_cls = nn.Sequential(
            nn.Conv3d(64, num_classes, kernel_size=(1, 1, 1))
        )

    def _make_layer(self, in_channels, out_channels, block_num, block=ConvBlock, kernel_size=(3, 3, 3),
                    stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.ReLU(inplace=True),
                get_conv(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0),
                            weight_std=self.weight_std)
            )
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(in_channels, out_channels, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))

        for i in range(1, block_num):
            layers.append(block(in_channels, out_channels, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)


    def forward(self, x_list):
        x, x_res = x_list
        x_0 = x[:, 0, :, :, :] # CT data
        x_1 = x[:, 1, :, :, :] # PET data

        x_0 = x_0.unsqueeze(1)
        x_1 = x_1.unsqueeze(1)

        # Encoder
        # Stage 1
        x_0 = self.encoder0(x_0)
        x_1 = self.encoder00(x_1)

        x_0 = self.layer0(x_0)
        x_1 = self.layer0(x_1)

        skip0 = torch.cat([x_0, x_1], dim=1)

        # Stage 2
        x_0 = self.encoder1(x_0)
        x_1 = self.encoder11(x_1)

        x_0 = self.layer1(x_0)
        x_1 = self.layer1(x_1)

        skip1 = torch.cat([x_0, x_1], dim=1)

        # Stage 3
        x_0 = self.encoder2(x_0)
        x_1 = self.encoder22(x_1)

        x_0 = self.layer2(x_0)
        x_1 = self.layer2(x_1)

        skip2 = torch.cat([x_0, x_1], dim=1)

        # Stage 4
        x_0 = self.encoder3(x_0)
        x_1 = self.encoder33(x_1)

        x_0 = self.layer3(x_0)
        x_1 = self.layer3(x_1)

        skip3 = torch.cat([x_0, x_1], dim=1)


        x_0 = self.layer4(x_0)
        x_1 = self.layer4(x_1)

        x = torch.cat([x_0, x_1], dim=1)
        x = self.fusionConv(x)

        # Decoder
        # res_x4 =
        return x


def COCONNET(input_size, num_classes=2, weight_std=True):
    # model = conresnet(shape, block=ConvBlock, layers=[1, 2, 2, 2, 2], num_classes=num_classes, weight_std=weight_std)
    model = CoConNet(input_size, block=ConvBlock, layers=[1, 4, 4, 4, 4], num_classes=num_classes, weight_std=weight_std)
    return model










