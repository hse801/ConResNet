import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import glob
import os
import os.path as osp
from models.ConResNet import ConResNet
import timeit
from tensorboardX import SummaryWriter
from utils import loss
from utils.engine import Engine
from math import ceil
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from scipy.ndimage.morphology import binary_erosion
from medpy import metric


class ConfusionMatrix:
    """
    Code copied and modified from
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/evaluation/metrics.py
    """
    def __init__(self, pred=None, label=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.pred = pred
        self.label = label
        self.smooth = 1e-3

    def compute(self):
        self.pred = self.pred
        self.label = self.label
        # self.pred = np.where(self.pred > 0.5, 1, 0)
        self.tp = ((self.pred != 0) * (self.label != 0)).sum()
        self.fp = ((self.pred != 0) * (self.label == 0)).sum()
        self.tn = ((self.pred == 0) * (self.label == 0)).sum()
        self.fn = ((self.pred == 0) * (self.label != 0)).sum()

        self.pred_empty = not np.any(self.pred)
        # self.pred_full = np.all(self.pred)
        self.label_empty = not np.any(self.label)
        # self.label_full = np.all(self.label)
        return self.label_empty, self.pred_empty

    def get_matrix(self):

        self.compute()

        return self.tp, self.fp, self.tn, self.fn

    # def get_existence(self):
    #
    #     self.compute()
    #
    #     return self.pred_empty, self.pred_full, self.label_empty, self.label_full


def recall(pred=None, label=None, confusion_matrix=None, nan_for_nonexisting=True, smooth=1e-3, **kwargs):
    # TP / (TP + FN)
    # same for sensitivity

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(pred=pred, label=label)

    label_empty, pred_empty = confusion_matrix.compute()
    # pred_empty, pred_full, label_empty, label_full = confusion_matrix.get_existence()

    if label_empty and pred_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    recall = float(tp / (tp + fn + smooth))
    # print(f'tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}')
    # print(f'recall = {recall}')

    return recall


def precision(pred=None, label=None, confusion_matrix=None, nan_for_nonexisting=True, smooth=1e-3, **kwargs):
    # TP / (TP + FP)

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(pred=pred, label=label)

    label_empty, pred_empty = confusion_matrix.compute()
    # pred_empty, pred_full, label_empty, label_full = confusion_matrix.get_existence()

    if label_empty and pred_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    precision = float(tp / (tp + fp + smooth))

    return precision


def f1_score(pred=None, label=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., smooth=1e-3, **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(pred=pred, label=label)

    label_empty, pred_empty = confusion_matrix.compute()
    # pred_empty, pred_full, label_empty, label_full = confusion_matrix.get_existence()

    if label_empty and pred_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    precision_ = precision(pred, label, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(pred, label, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_ + smooth)


def hausdorff_distance(pred=None, label=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, smooth = 1e-3, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(pred, label)

    label_empty, pred_empty = confusion_matrix.compute()
    # pred_empty, pred_full, label_empty, label_full = confusion_matrix.get_existence()

    if label_empty or pred_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    pred, label = confusion_matrix.pred, confusion_matrix.label

    return metric.hd(pred, label, voxel_spacing, connectivity)


def hausdorff_distance_95(pred=None, label=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, smooth = 1e-3, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(pred, label)

    label_empty, pred_empty = confusion_matrix.compute()
    # pred_empty, pred_full, label_empty, label_full = confusion_matrix.get_existence()

    if label_empty or pred_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    pred, label = confusion_matrix.pred, confusion_matrix.label

    return metric.hd95(pred, label, voxel_spacing, connectivity)


def specificity(pred=None, label=None, confusion_matrix=None, nan_for_nonexisting=True, smooth=1e-3, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(pred, label)

    label_empty, pred_empty = confusion_matrix.compute()
    # pred_empty, pred_full, label_empty, label_full = confusion_matrix.get_existence()

    if label_empty and pred_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float(tn / (tn + fp + smooth))


# def precision_recall(pred, label):
#     precision, recall, thresholds = precision_recall_curve(label, pred)
#     precision = np.fliplr([precision])[0] # So the array is increasing(No negative AUC)
#     recall = np.fliplr([recall])[0] # So the array is increasing (No negative AUC)
#
#     AUC_prec_rec = np.trapz(precision, recall)
#     print(f'Area under Precision-Recall curve: {AUC_prec_rec}')
#     prec_rec_curve
# F1 score
# def compute_f1_score(pred, label):
#     pred = flatten_last(pred)
#     label = flatten_last(label)
#     pred = pred.cpu().numpy()
#     label = label.cpu().numpy()
#     # Conver to binary classification
#     pred = np.where(pred > 0.5, 1, 0)
#     # print(f'f1 pred.size() = {pred.size()}, label.size() = {label.size()}')
#     # f1 pred.size() = torch.Size([1638400, 2]), label.size() = torch.Size([1638400, 2])
#     label_primary = label[:, 0]
#     pred_primary = pred[:, 0]
#
#     label_lymph = label[:, 1]
#     pred_lymph = pred[:, 1]
#
#     F1_score_p = f1_score(label_primary, pred_primary, labels=None, average='binary', sample_weight=None)
#     print(f'F1 score (Primary): {F1_score_p:.4f}')
#     F1_score_l = f1_score(label_lymph, pred_lymph, labels=None, average='binary', sample_weight=None)
#     print(f'F1 score (Lymph): {F1_score_l:.4f}')
#     return F1_score_p, F1_score_l


def flatten_last(tensor):
    """
    flattens a given tensor such that the channel axis is last
    for the f1 score function
    (N, C, D, H, W) --> (N * D * H * W, C)
    """
    # C: number of channels
    C = tensor.size(1)
    # New axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) --> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) --> (C, N * D * H * W)
    return transposed.contiguous().view(-1, C)


def flatten(tensor):
    """
    flattens a given tensor such that the channel axis is first
    (N, C, D, H, W) --> (C, N * D * H * W)
    """
    # C: number of channels
    C = tensor.size(1)
    # New axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) --> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) --> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_channel_dice(input, target, epsilon=1e-6, weight=None):
    # print(f'bf input.size() = {input.size()}, target.size() = {target.size()}')
    # (N, C, D, H, W) --> N: Batch size, C: channel(class number)
    # input.size() = torch.Size([2, 2, 80, 128, 160]), target.size() = torch.Size([2, 2, 80, 128, 160])
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # primary_nonzero = target[:, 0, :, :, :].nonzero()
    # lymph_nonzero = target[:, 1, :, :, :].nonzero()
    # print(f'primary_nonzero = {primary_nonzero.size()}, primary_nonzero = {primary_nonzero}')
    # print(f'lymph_nonzero = {lymph_nonzero.size()}, lymph_nonzero = {lymph_nonzero}')
    # if primary_nonzero.nelement() == 0:
    #     print(f'No primary tumor')
    #     input = input[:, 1, :, :, :]
    #     target = target[:, 1, :, :, :]
    #     input = input.unsqueeze(1)
    #     target = target.unsqueeze(1)
    # if lymph_nonzero.nelement() == 0:
    #     print(f'No lymph node')
    #     input = input[:, 0, :, :, :]
    #     target = target[:, 0, :, :, :]
    #     input = input.unsqueeze(1)
    #     target = target.unsqueeze(1)
    # print(f'bf input.size() = {input.size()}, target.size() = {target.size()}')
    # input = F.sigmoid(input)
    input = flatten(input)
    target = flatten(target)
    target = target.float()
    # print(f'af input.size() = {input.size()}, target.size() = {target.size()}')

    # intersect = torch.sum(torch.mul(input, target), dim=1)
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # denominator = torch.sum(input, dim=1) + torch.sum(target, dim=1) + 1e-3
    denominator = (input + target).sum(-1) + 1e-3
    dice = 2 * intersect / denominator

    #print(f'target max = {torch.max(target)}, input = {torch.max(input)}')
    # print(f'intersect = {intersect}, denom = {denominator}')
    # print(f'dice = {dice}, shape = {dice.size()}')
    return dice.cpu().data.numpy()


# def hausdorff_distance(pred, target):
#     """Code copied from
#     https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
#     for compatibility reason with python 3.6
#     """
#     a_points = np.transpose(np.nonzero(pred.cpu().numpy()))
#     b_points = np.transpose(np.nonzero(target.cpu().numpy()))
#
#     # Handle empty sets properly:
#     # - if both sets are empty, return zero
#     # - if only one set is empty, return infinity
#     if len(a_points) == 0:
#         return 0 if len(b_points) == 0 else np.inf
#     elif len(b_points) == 0:
#         return np.inf
#
#     return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
#                max(cKDTree(b_points).query(a_points, k=1)[0]))


# def compute_channel_hausdorff(pred, target):
#     assert pred.size() == target.size(), "'pred' and 'target' must have the same shape"
#     pred = F.sigmoid(pred)
#     pred = pred.squeeze()
#     target = target.squeeze()
#     # pred = flatten(pred)
#     # target = flatten(target)
#     # target = target.float()
#     print(f'pred size = {pred.size()}, target = {target.size()}')
#     hausdorff_dist = directed_hausdorff(pred.cpu().numpy(), target.cpu().numpy())
#     # hausdorff_dist_1 = directed_hausdorff(pred[0, :].cpu().numpy(), target[0, :].cpu().numpy())[0]
#     # hausdorff_dist_2 = directed_hausdorff(pred[1, :].cpu().numpy(), target[1, :].cpu().numpy())[0]
#     # hausdorff_dist = [hausdorff_dist_1, hausdorff_dist_2]
#     # print(f'hausdorff_dist_1 = {hausdorff_dist_1}, hausdorff_dist_2 = {hausdorff_dist_2}')
#
#     return hausdorff_dist


def dice_score2(input, labels):
    input = F.sigmoid(input)

    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * labels).sum(dim=axes)
    union = bin_input.sum(dim=axes) + labels.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def compute_dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = F.sigmoid(preds)

    pred = preds[:, 0, :, :, :]
    label = labels[:, 0, :, :, :]
    dice = dice_score(pred, label).cpu().data.numpy()
    # print(f'dice = {dice}')
    return dice


def predict_sliding(net, imagelist, tile_size, classes):
    """
    strideHW = 86, strideD = 54
    tile_deps = 1, tile_rows = 1, tile_cols = 1
    d1 = 0, d2 = 80, x1 = 0, x2 = 160, y1 = 0, y2 = 128
    """
    # print('predict sliding')
    image, image_res = imagelist
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs).cuda()
    count_predictions = torch.from_numpy(count_predictions).cuda()
    # print(f'strideHW = {strideHW}, strideD = {strideD}')
    # print(f'tile_deps = {tile_deps}, tile_rows = {tile_rows}, tile_cols = {tile_cols}')


    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                # print(f'd1 = {d1}, d2 = {d2}, x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}')

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                prediction = net([img, img_res])
                prediction = prediction[0]

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction
                # print(f'count_predictions = {count_predictions.size()}, full_probs = {full_probs}')

    full_probs /= count_predictions
    # print(f'predict sliding: full_probs size = {full_probs.size()}')
    return full_probs
