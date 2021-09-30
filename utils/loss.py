import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from math import ceil


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2 * num / den
        loss_avg = 1 - dice_score.mean()

        return loss_avg


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict %s & target %s shape do not match' % (predict.shape, target.shape)
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.sigmoid(predict)
        # Get loss except empty label
        primary_nonzero = target[:, 0, :, :, :].nonzero()
        lymph_nonzero = target[:, 1, :, :, :].nonzero()

        if primary_nonzero.nelement() == 0:
            self.ignore_index = 0
        elif lymph_nonzero.nelement() == 0:
            self.ignore_index = 1
        else:
            self.ignore_index = None

        # print(f'self.ignore_index = {self.ignore_index}, target shape = {target.shape[1]}')
        # self.ignore_index = None, target shape = 2
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                # print(f'predict shape = {predict.shape}, target = {target.shape}')
                # print(f'shape predict[:, i] = {np.shape(predict[:, i])}, target[:, i] = {np.shape(target[:, i])}')
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    print(f'weight is not None')
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/(target.shape[1]-1 if self.ignore_index != None else target.shape[1])


class BCELoss(nn.Module):
    def __init__(self, ignore_index=None, **kwargs):
        super(BCELoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights = None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-7, max=1-1e-7)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        # Get loss except empty label
        primary_nonzero = target[:, 0, :, :, :].nonzero()
        lymph_nonzero = target[:, 1, :, :, :].nonzero()

        if primary_nonzero.nelement() == 0:
            self.ignore_index = 0
        elif lymph_nonzero.nelement() == 0:
            self.ignore_index = 1
        else:
            self.ignore_index = None

        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss

        return total_loss.mean()


class BCELossBoud(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(BCELossBoud, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-3, max=1-1e-3)
            bce = weights[1] * (target * torch.log(output)) + \
                  weights[0] * ((1-target) * torch.log((1-output)))
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1-target) * torch.log((1-output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):

        bs, category, depth, width, height = target.shape
        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:, i]
            targ_i = target[:, i]
            tt = np.log(depth * width * height / (target[:, i].cpu().data.numpy().sum()+1))
            bce_i = self.weighted_BCE_cross_entropy(pred_i, targ_i, weights=[1, tt])
            bce_loss.append(bce_i)

        bce_loss = torch.stack(bce_loss)
        total_loss = bce_loss.mean()
        # print(f'loss.py: bce_loss = {bce_loss}, total_loss = {total_loss}')
        return total_loss
