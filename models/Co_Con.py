
import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import scipy.ndimage as ndimage


class ConvStd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(ConvStd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        # print(f'weight bf = {weight.size()}')
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        # print(f'x size = {x.size()}, weight = {weight.size()}')
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_conv(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), bias=False,
             weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        # print(f'weight std = {weight_std}')
        # print(f'in_planes = {in_planes},out_planes = {out_planes} ')
        return ConvStd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                       bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


class AttBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), bias=False, weight_std=False, first_layer=False):
        super(AttBlock, self).__init__()
        self.weight_std = weight_std
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.first_layer = first_layer

        self.relu = nn.ReLU(inplace=True)
        print(f'for group norm in = {in_planes}, out planes = {out_planes}')
        self.gn_seg = nn.GroupNorm(8, in_planes)
        self.conv_seg = get_conv(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                 stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                                 dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_res = nn.GroupNorm(8, out_planes)
        self.conv_res = get_conv(out_planes, out_planes, kernel_size=(1, 1, 1),
                                 stride=(1, 1, 1), padding=(0,0,0),
                                 dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_res1 = nn.GroupNorm(8, out_planes)
        self.conv_res1 = get_conv(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                  stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                  dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.gn_res2 = nn.GroupNorm(8, out_planes)
        self.conv_res2 = get_conv(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                  stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                  dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_mp = nn.GroupNorm(8, in_planes)
        # modify channel 4 -> 2
        self.conv_mp_first = get_conv(2, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                      stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                                      dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.conv_mp = get_conv(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

    def _res(self, x):  # bs, channel, D, W, H

        bs, channel, depth, heigt, width = x.shape
        x_copy = torch.zeros_like(x).cuda()
        x_copy[:, :, 1:, :, :] = x[:, :, 0: depth - 1, :, :]
        res = x - x_copy
        res[:, :, 0, :, :] = 0
        res = torch.abs(res)

        # print(f'x.shape = {x.shape}, type = {type(x)}')
        # res = np.zeros(bs, channel, depth, heigt, width)
        # res = torch.zeros_like(x).cuda()
        # x = x.detach().cpu().numpy()
        # res = res.detach().cpu().numpy()
        # # print(f'res.shape = {res.shape}, x type = {type(x)}')
        # for b in range(bs):
        #     for c in range(channel):
        #         x_slice = x[b, c, :, :, :]
        #         sx = ndimage.sobel(x_slice, axis=0, mode='constant')
        #         sy = ndimage.sobel(x_slice, axis=1, mode='constant')
        #         res[b, c, :, :, :] = np.hypot(sx, sy)
        #         res[b, c, :, :, :] = (res[b, c, :, :, :] - np.min(res[b, c, :, :, :])) / (np.max(res[b, c, :, :, :]) - np.min(res[b, c, :, :, :]))
        # res = torch.from_numpy(res).float().cuda()
        # print(f'res type = {type(res)}')

        return res

    def forward(self, input):
        x1, x2 = input
        if self.first_layer:
            # print(f'first layer: x1 = {x1.size()}, x2 = {x2.size()}')
            # first layer: x1 = torch.Size([1, 64, 20, 32, 40]), x2 = torch.Size([1, 2, 20, 32, 40])
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)
            # print(f'AttBlock x2 = {x2.size()}, res = {res.size()}')
            #
            x2 = self.conv_mp_first(x2)
            # print(f'2ConResAtt x2 = {x2.size()}, res = {res.size()}')
            x2 = x2 + res

        else:
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)

            if self.in_planes != self.out_planes:
                x2 = self.gn_mp(x2)
                x2 = self.relu(x2)
                x2 = self.conv_mp(x2)

            x2 = x2 + res

        x2 = self.gn_res1(x2)
        x2 = self.relu(x2)
        x2 = self.conv_res1(x2)

        x1 = x1 * (1 + torch.sigmoid(x2))

        return [x1, x2]


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=(1, 1, 1), dilation=(1, 1, 1), downsample=None, fist_dilation=1,
                 multi_grid=1, weight_std=False):
        super(ConvBlock, self).__init__()
        self.weight_std = weight_std
        self.relu = nn.ReLU(inplace=True)

        self.gn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = get_conv(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=dilation * multi_grid,
                              dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.gn2 = nn.GroupNorm(8, planes)
        self.conv2 = get_conv(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=dilation * multi_grid,
                              dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        skip = x

        seg = self.gn1(x)
        seg = self.relu(seg)
        seg = self.conv1(seg)

        seg = self.gn2(seg)
        seg = self.relu(seg)
        seg = self.conv2(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class ASPP(nn.Module):
    """
    3D Astrous Spatial Pyramid Pooling
    """
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.pool = nn.MaxPool3d(3)
        self.aspp_block1 = nn.Sequential(
            nn.Conv3d(
                in_dims, out_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(rate[0], rate[0], rate[0]),
                dilation=(rate[0], rate[0], rate[0])
            ),
            nn.PReLU(),
            # nn.BatchNorm3d(out_dims),
            nn.GroupNorm(8, out_dims)
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv3d(
                in_dims, out_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(rate[1], rate[1], rate[1]),
                dilation=(rate[1], rate[1], rate[1])
            ),
            nn.PReLU(),
            # nn.BatchNorm3d(out_dims),
            nn.GroupNorm(8, out_dims)
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv3d(
                in_dims, out_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(rate[2], rate[2], rate[2]),
                dilation=(rate[2], rate[2], rate[2])
            ),
            nn.PReLU(),
            # nn.BatchNorm3d(out_dims),
            nn.GroupNorm(8, out_dims)
        )

        self.output = nn.Conv3d(len(rate) * out_dims, out_dims, kernel_size=(1, 1, 1))
        self._init_weights()

    def forward(self, x):
        x = self.pool(x)
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CoConNet(nn.Module):
    def __init__(self, shape, block, layers, num_filters=24, in_channels=1, num_classes=2, weight_std=False):
        super(CoConNet, self).__init__()
        self.shape = shape
        self.weight_std = weight_std
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.block = block
        self.layers = layers
        self.size_divide = 4
        # print(f'weight std = {weight_std}')
        self.encoder0 = nn.Sequential(
            # Input channel -> num_filters
            # 2 -> 32
            get_conv(in_planes=in_channels, out_planes=self.num_filters, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                     padding=(1, 1, 1), bias=False)
        )
        self.encoder00 = nn.Sequential(
            get_conv(in_planes=in_channels, out_planes=self.num_filters, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                     padding=(1, 1, 1), bias=False)
        )
        self.encoder1 = nn.Sequential(
            # 32 -> 64
            nn.GroupNorm(8, self.num_filters),
            get_conv(in_planes=self.num_filters, out_planes=self.num_filters * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                     padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 2)
        )
        self.encoder11 = nn.Sequential(
            nn.GroupNorm(8, self.num_filters),
            get_conv(in_planes=self.num_filters, out_planes=self.num_filters * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                     padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 2)
        )
        self.encoder2 = nn.Sequential(
            # 64 -> 128
            nn.GroupNorm(8, self.num_filters * 2),
            get_conv(in_planes=self.num_filters * 2, out_planes=self.num_filters * 4, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                     padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 4)
        )
        self.encoder22 = nn.Sequential(
            nn.GroupNorm(8, self.num_filters * 2),
            get_conv(in_planes=self.num_filters * 2, out_planes=self.num_filters * 4, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                     padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 4)
        )
        self.encoder3 = nn.Sequential(
            # 128 -> 256
            nn.GroupNorm(8, self.num_filters * 4),
            get_conv(in_planes=self.num_filters * 4, out_planes=self.num_filters * 8, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                     padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 8)
        )
        self.encoder33 = nn.Sequential(
            nn.GroupNorm(8, self.num_filters * 4),
            get_conv(in_planes=self.num_filters * 4, out_planes=self.num_filters * 8, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                     padding=(1, 1, 1), bias=False),
            nn.PReLU(self.num_filters * 8)
        )

        # layers=[1, 2, 2, 2, 2]
        self.layer0 = self._make_layer(block, self.num_filters, self.num_filters, layers[0], stride=(1, 1, 1))
        self.layer0_2 = self._make_layer(block, self.num_filters, self.num_filters, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, self.num_filters * 2, self.num_filters * 2, layers[1], stride=(1, 1, 1))
        self.layer1_2 = self._make_layer(block, self.num_filters * 2, self.num_filters * 2, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, self.num_filters * 4, self.num_filters * 4, layers[2], stride=(1, 1, 1))
        self.layer2_2 = self._make_layer(block, self.num_filters * 4, self.num_filters * 4, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, self.num_filters * 8, self.num_filters * 8, layers[3], stride=(1, 1, 1))
        self.layer3_2 = self._make_layer(block, self.num_filters * 8, self.num_filters * 8, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(block, self.num_filters * 8, self.num_filters * 8, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2))
        self.layer4_2 = self._make_layer(block, self.num_filters * 8, self.num_filters * 8, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2))
        # self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        # self.layer0_2 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        # self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        # self.layer1_2 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        # self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        # self.layer2_2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        # self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        # self.layer3_2 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        # self.layer4 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2))
        # self.layer4_2 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, self.num_filters * 8),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            get_conv(self.num_filters * 8, self.num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), weight_std=self.weight_std)
        )
        self.fusionConv_2 = nn.Sequential(
            nn.GroupNorm(8, self.num_filters * 8),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            get_conv(self.num_filters * 8, self.num_filters * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), weight_std=self.weight_std)
        )
        self.aspp1 = nn.Sequential(
            ASPP(self.num_filters * 8, self.num_filters * 4)
        )
        self.aspp2 = nn.Sequential(
            ASPP(self.num_filters * 8, self.num_filters * 4)
        )

        self.seg_x4 = nn.Sequential(
            AttBlock(self.num_filters * 8, self.num_filters * 4, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        self.seg_x2 = nn.Sequential(
            AttBlock(self.num_filters * 4, self.num_filters * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))
        self.seg_x1 = nn.Sequential(
            AttBlock(self.num_filters * 2, self.num_filters * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))

        self.seg_cls = nn.Sequential(
            nn.Conv3d(self.num_filters * 2, num_classes, kernel_size=1)
        )
        self.res_cls = nn.Sequential(
            nn.Conv3d(self.num_filters * 2, num_classes, kernel_size=1)
        )
        self.resx2_cls = nn.Sequential(
            nn.Conv3d(self.num_filters * 2, num_classes, kernel_size=1)
        )
        self.resx4_cls = nn.Sequential(
            nn.Conv3d(self.num_filters * 4, num_classes, kernel_size=1)
        )

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.ReLU(inplace=True),
                get_conv(inplanes, outplanes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0),
                         weight_std=self.weight_std)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, outplanes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        for i in range(1, blocks):
            layers.append(
                block(inplanes, outplanes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))
        return nn.Sequential(*layers)

    def forward(self, x_list):
        x, x_res = x_list
        # print(f'x size = {x.size()}, x_res size = {x_res.size()}')

        """
        [Batch, Channel, Depth, Width, Height]
        x size = torch.Size([1, 2, 80, 128, 160]), x_res size = torch.Size([1, 2, 80, 128, 160])
        x1 = torch.Size([1, 32, 80, 128, 160])
        x2 = torch.Size([1, 64, 40, 64, 80])
        x3 = torch.Size([1, 128, 20, 32, 40])
        x4 = torch.Size([1, 256, 10, 16, 20])
        x5 = torch.Size([1, 256, 10, 16, 20])
        x6 = torch.Size([1, 128, 10, 16, 20])
        """
        x, x_res = x_list
        x_0 = x[:, 0, :, :, :] # CT data
        x_1 = x[:, 1, :, :, :] # PET data

        x_0 = x_0.unsqueeze(1)
        x_1 = x_1.unsqueeze(1)

        # Encoder
        # Stage 1
        # print(f'x_0 = {x_0.size()}, x_1 = {x_1.size()}')
        # x_0 = torch.Size([1, 1, 80, 128, 160]), x_1 = torch.Size([1, 1, 80, 128, 160])
        x_0 = self.encoder0(x_0)
        x_1 = self.encoder00(x_1)
        # print(f'bf layer 0: x0 = {x_0.size()}, x_1 = {x_1.size()}')
        # bf layer 0: x0 = torch.Size([1, 16, 80, 128, 160]), x_1 = torch.Size([1, 16, 80, 128, 160])
        x_0 = self.layer0(x_0)
        x_1 = self.layer0(x_1)
        # print(f'af layer 0: x0 = {x_0.size()}, x_1 = {x_1.size()}')

        skip1 = torch.cat([x_0, x_1], dim=1)
        # print(f'skip1 = {skip1.size()}')
        # skip1 = torch.Size([1, 16, 80, 128, 160])

        # Stage 2
        x_0 = self.encoder1(x_0)
        x_1 = self.encoder11(x_1)
        # print(f'bf layer 1: x0 = {x_0.size()}, x_1 = {x_1.size()}')
        # bf layer 1: x0 = torch.Size([1, 16, 80, 128, 160]), x_1 = torch.Size([1, 16, 80, 128, 160])
        x_0 = self.layer1(x_0)
        x_1 = self.layer1(x_1)
        # print(f'af layer 1: x0 = {x_0.size()}, x_1 = {x_1.size()}')
        # af layer 1: x0 = torch.Size([1, 16, 80, 128, 160]), x_1 = torch.Size([1, 16, 80, 128, 160])
        skip2 = torch.cat([x_0, x_1], dim=1)
        # print(f'skip2 = {skip2.size()}')
        # skip2 = torch.Size([1, 32, 40, 64, 80])

        # Stage 3
        x_0 = self.encoder2(x_0)
        x_1 = self.encoder22(x_1)

        x_0 = self.layer2(x_0)
        x_1 = self.layer2(x_1)
        # print(f'bf skip2 x_0 = {x_0.size()}, x1 = {x_1.size()}')
        # bf skip2 x_0 = torch.Size([1, 32, 20, 32, 40]), x1 = torch.Size([1, 32, 20, 32, 40])
        skip3 = torch.cat([x_0, x_1], dim=1)
        # print(f'skip3 = {skip3.size()}')
        # skip3 = torch.Size([1, 64, 20, 32, 40])

        # Stage 4
        x_0 = self.encoder3(x_0)
        x_1 = self.encoder33(x_1)

        x_0 = self.layer3(x_0)
        x_1 = self.layer3(x_1)

        # skip3 = torch.cat([x_0, x_1], dim=1)
        # print(f'skip3 = {skip3.size()}')
        # skip3 = torch.Size([1, 128, 10, 16, 20])

        # print(f'bf layer4 x_0 = {x_0.size()}, x1 = {x_1.size()}')
        x_0 = self.layer4(x_0)
        x_1 = self.layer4(x_1)
        # print(f'af layer4 x_0 = {x_0.size()}, x1 = {x_1.size()}')
        # af layer4 x_0 = torch.Size([1, 64, 10, 16, 20]), x1 = torch.Size([1, 64, 10, 16, 20])

        # print(f'bf fusionConv x = {x.size()}')
        # bf fusionConv x = torch.Size([1, 128, 10, 16, 20])

        # x_0 = self.fusionConv(x_0)
        # x_1 = self.fusionConv_2(x_1)

        #bf aspp x_0 = torch.Size([1, 192, 10, 16, 20]), x1 = torch.Size([1, 192, 10, 16, 20])
        #af aspp x_0 = torch.Size([1, 96, 3, 5, 6]), x1 = torch.Size([1, 96, 3, 5, 6])

        x_0 = self.fusionConv(x_0)
        x_1 = self.fusionConv_2(x_1)


        x = torch.cat([x_0, x_1], dim=1)

        res_x4 = F.interpolate(x_res, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        seg_x4 = F.interpolate(x, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        # print(f'1 seg x4 = {seg_x4.size()}, skip3 = {skip3.size()}, resx4 = {res_x4.size()}')
        # 1 seg x4 = torch.Size([1, 64, 20, 32, 40]), skip3 = torch.Size([1, 64, 20, 32, 40]), resx4 = torch.Size([1, 2, 20, 32, 40])
        seg_x4 = seg_x4 + skip3
        # print(f'seg x4 = {seg_x4.size()}, skip3 = {skip3.size()}, resx4 = {res_x4.size()}')
        # seg x4 = torch.Size([1, 64, 20, 32, 40]), skip3 = torch.Size([1, 64, 20, 32, 40]), resx4 = torch.Size([1, 2, 20, 32, 40])
        seg_x4, res_x4 = self.seg_x4([seg_x4, res_x4])
        # print(f'af seg x4 = {seg_x4.size()}, res x4 = {res_x4.size()}')

        res_x2 = F.interpolate(res_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = F.interpolate(seg_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = seg_x2 + skip2
        seg_x2, res_x2 = self.seg_x2([seg_x2, res_x2])

        res_x1 = F.interpolate(res_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = F.interpolate(seg_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        # print(f'seg_x1 = {seg_x1.size()}')
        # seg_x1 = torch.Size([1, 32, 80, 128, 160])
        seg_x1 = seg_x1 + skip1
        seg_x1, res_x1 = self.seg_x1([seg_x1, res_x1])

        seg = self.seg_cls(seg_x1)
        res = self.res_cls(res_x1)
        resx2 = self.resx2_cls(res_x2)
        resx4 = self.resx4_cls(res_x4)

        resx2 = F.interpolate(resx2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                      mode='trilinear', align_corners=True)
        resx4 = F.interpolate(resx4, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                      mode='trilinear', align_corners=True)

        return [seg, res, resx2, resx4]


def Co_Con(shape, num_classes=2, weight_std=True):
    """
    Co-Learning and Context Network without ASPP block
    :param shape:
    :param num_classes:
    :param weight_std:
    :return:
    """
    # model = CoConNet(shape, block=ConvBlock, layers=[1, 1, 1, 1, 1], num_classes=num_classes, weight_std=weight_std)
    model = CoConNet(shape, block=ConvBlock, layers=[1, 2, 2, 2, 2], num_classes=num_classes, weight_std=weight_std)
    # model = CoConNet(input_size, block=ConvBlock, layers=[1, 4, 4, 4, 4], num_classes=num_classes, weight_std=weight_std)
    # model = CoConNet(shape, block=ConvBlock, layers=[1, 4, 4, 4, 4], num_classes=num_classes, weight_std=weight_std)
    return model
