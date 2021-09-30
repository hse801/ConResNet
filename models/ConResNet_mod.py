import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import scipy.ndimage as ndimage


class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        # print(f'ConResNet.py: ConvStd: weight bf = {weight.size()}')
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        # print(f'ConResNet.py: ConvStd: x size = {x.size()}, weight = {weight.size()}')
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), bias=False,
              weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        # print(f'Conresnet_mod: Is weight std')
        # print(f'weight std = {weight_std}')
        # print(f'in_planes = {in_planes},out_planes = {out_planes} ')
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


class ConResAtt(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), bias=False, weight_std=False, first_layer=False):
        super(ConResAtt, self).__init__()
        self.weight_std = weight_std
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.first_layer = first_layer
        print(f'1weight_std = {weight_std}')

        self.relu = nn.ReLU(inplace=True)

        self.gn_seg = nn.GroupNorm(8, in_planes)
        self.conv_seg = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        # print(f'kernel size [0] = {kernel_size}, {kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]}')
        # print(f'stride size [0] = {stride[0]}, {stride[1]}, {stride[2]}')
        # print(f'padding size [0] = {padding[0]}, {padding[1]}, {padding[2]}')

        self.gn_res = nn.GroupNorm(8, out_planes)
        self.conv_res = conv3x3x3(out_planes, out_planes, kernel_size=(1,1,1),
                               stride=(1, 1, 1), padding=(0,0,0),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        print(f'2weight_std = {weight_std}')

        self.gn_res1 = nn.GroupNorm(8, out_planes)
        self.conv_res1 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.gn_res2 = nn.GroupNorm(8, out_planes)
        self.conv_res2 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_mp = nn.GroupNorm(8, in_planes)
        # modify channel 4 -> 2
        # print(f'kernel size [0] = {kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]}')
        # print(f'stride size [0] = {stride[0]}, {stride[1]}, {stride[2]}')
        # print(f'padding size [0] = {padding[0]}, {padding[1]}, {padding[2]}')
        self.conv_mp_first = conv3x3x3(2, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                              stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                              dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        print(f'weight_std = {weight_std}')
        self.conv_mp = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

    def _res(self, x):  # bs, channel, D, W, H

        bs, channel, depth, heigt, width = x.shape
        x_copy = torch.zeros_like(x).cuda()
        x_copy[:, :, 1:, :, :] = x[:, :, 0: depth - 1, :, :]
        res = x - x_copy
        res[:, :, 0:1, :, :] = 0
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
            print(f'first layer: x1 = {x1.size()}, x2 = {x2.size()}')
            # first layer: x1 = torch.Size([1, 128, 20, 32, 40]), x2 = torch.Size([1, 2, 20, 32, 40])
            x1 = self.gn_seg(x1)
            x1 = self.relu(x1)
            x1 = self.conv_seg(x1)

            res = torch.sigmoid(x1)
            res = self._res(res)
            res = self.conv_res(res)
            print(f'first x2 = {x2.size()}, res = {res.size()}')
            # first x2 = torch.Size([1, 2, 20, 32, 40]), res = torch.Size([1, 64, 20, 32, 40])
            x2 = self.conv_mp_first(x2)
            print(f'2first x2 = {x2.size()}, res = {res.size()}')
            # 2first x2 = torch.Size([1, 64, 20, 32, 40]), res = torch.Size([1, 64, 20, 32, 40])
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

        x1 = x1*(1 + torch.sigmoid(x2))

        return [x1, x2]


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=(1, 1, 1), dilation=(1, 1, 1), downsample=None, fist_dilation=1,
                 multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.relu = nn.ReLU(inplace=True)
        print(f'dilation = {dilation}, type dilation = {type(dilation)}, multi grid = {type(multi_grid)}')
        print(f'inplanes = {inplanes}, planes = {planes}')

        self.gn1 = nn.GroupNorm(8, inplanes)
        print(f'padding = {dilation * multi_grid}, self.weight_std = {self.weight_std}')
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.gn2 = nn.GroupNorm(8, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=dilation * multi_grid,
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


class conresnet(nn.Module):
    def __init__(self, shape, block, layers, num_classes=1, weight_std=False):
        self.shape = shape
        self.weight_std = weight_std
        super(conresnet, self).__init__()

        self.conv_4_32 = nn.Sequential(
            # modify channel 4 -> 2
            conv3x3x3(2, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_std=self.weight_std))

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            conv3x3x3(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            conv3x3x3(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            conv3x3x3(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        # layers=[1, 2, 2, 2, 2]
        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), weight_std=self.weight_std)
        )

        self.seg_x4 = nn.Sequential(
            ConResAtt(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        self.seg_x2 = nn.Sequential(
            ConResAtt(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))
        self.seg_x1 = nn.Sequential(
            ConResAtt(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))

        self.seg_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.res_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.resx2_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.resx4_cls = nn.Sequential(
            nn.Conv3d(64, num_classes, kernel_size=1)
        )

    def _make_layer(self, block, inplanes, outplanes, blocks, stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.ReLU(inplace=True),
                conv3x3x3(inplanes, outplanes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0),
                            weight_std=self.weight_std)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, outplanes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        print(f'multi_grid = {multi_grid}')
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
        ## encoder
        x = self.conv_4_32(x)
        # print(f'x1 = {x.size()}')
        residual = x
        print(f'bf layer 0: x = {x.size()}')
        x = self.layer0(x)
        x += residual
        skip1 = x
        # print(f'skip1 = {skip1.size()}')
        # skip1 = torch.Size([1, 32, 80, 128, 160])
        # print(f'x2 = {x.size()}')

        x = self.conv_32_64(x)
        # print(f'bf layer1 x = {x.size()}')
        # bf layer1 x = torch.Size([1, 64, 40, 64, 80])
        residual = x
        x = self.layer1(x)
        x += residual
        skip2 = x
        # print(f'skip2 = {skip2.size()}')
        # skip2 = torch.Size([1, 64, 40, 64, 80])

        x = self.conv_64_128(x)
        residual = x
        x = self.layer2(x)
        x += residual
        skip3 = x
        # print(f'skip3 = {skip3.size()}')
        # skip3 = torch.Size([1, 128, 20, 32, 40])

        x = self.conv_128_256(x)
        residual = x
        x = self.layer3(x)
        x += residual
        # print(f'bf layer 4 x = {x.size()}')
        # bf layer 4 x = torch.Size([1, 256, 10, 16, 20])
        x = self.layer4(x)
        # print(f'bf fusion conv x = {x.size()}')
        # bf fusion conv x = torch.Size([1, 256, 10, 16, 20])
        x = self.fusionConv(x)
        # print(f'final x = {x.size()}')
        # final x = torch.Size([1, 128, 10, 16, 20])

        ## decoder
        # print(f'xres = {x_res.size()}, x = {x.size()}')
        # xres = torch.Size([1, 2, 80, 128, 160]), x = torch.Size([1, 128, 10, 16, 20])
        res_x4 = F.interpolate(x_res, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        seg_x4 = F.interpolate(x, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        seg_x4 = seg_x4 + skip3
        print(f'seg x4 = {seg_x4.size()}, skip3 = {skip3.size()}, resx4 = {res_x4.size()}')
        # segx4 = torch.Size([1, 128, 20, 32, 40]), skip3 = torch.Size([1, 128, 20, 32, 40]), resx4 = torch.Size([1, 2, 20, 32, 40])
        seg_x4, res_x4 = self.seg_x4([seg_x4, res_x4])


        res_x2 = F.interpolate(res_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = F.interpolate(seg_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = seg_x2 + skip2
        seg_x2, res_x2 = self.seg_x2([seg_x2, res_x2])

        res_x1 = F.interpolate(res_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = F.interpolate(seg_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
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
        # print(f'ConResNet.py: seg = {seg.size()}, res = {res.size()}, resx2 = {resx2.size()}, resx4 = {resx4.size()}')
        # seg = torch.Size([4, 2, 80, 128, 160]), res = torch.Size([4, 2, 80, 128, 160]),
        # resx2 = torch.Size([4, 2, 80, 128, 160]), resx4 = torch.Size([4, 2, 80, 128, 160])
        return [seg, res, resx2, resx4]


def ConResNet_mod(shape, num_classes=1, weight_std=True):

    # model = conresnet(shape, block=ConvBlock, layers=[1, 2, 2, 2, 2], num_classes=num_classes, weight_std=weight_std)
    model = conresnet(shape, block=NoBottleneck, layers=[1, 4, 4, 4, 4], num_classes=num_classes, weight_std=weight_std)
    return model
