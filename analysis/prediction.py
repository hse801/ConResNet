import os
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import glob
import torch
import dataset
from models.ConResNet import ConResNet
from models.ConResNet_mod import ConResNet_mod
from models.Co_Con import Co_Con
import utils.metrics
from utils.metrics import ConfusionMatrix
from utils.metrics import compute_channel_dice
from math import ceil
import torch.nn.functional as F
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from scipy.ndimage.morphology import binary_erosion



def predictor(PATH, data_loader):

    # 2 channel label, 2 classes
    model_path = PATH + '/ConResNet_BEST.pth'
    path_list = glob.glob('F:/LungCancerData/valid/*/')
    input_size = (80, 128, 160)
    model = Co_Con(input_size, num_classes=2, weight_std=True)
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    model.eval()

    val = 0.0
    primary_dice = []
    lymph_dice = []
    f1_p_list = []
    f1_l_list = []

    eval_list = []

    for index, batch in enumerate(data_loader):
        # print('%d processd'%(index))
        # image, image_res, label, size, name, affine = batch
        image, image_res, label = batch
        image = image.cuda()
        image_res = image_res.cuda()
        label = label.cuda()
        with torch.no_grad():
            # pred = utils.metrics.predict_sliding(model, [image, image_res], input_size, classes=2)
            """
            strideHW = 86, strideD = 54
            tile_deps = 1, tile_rows = 1, tile_cols = 1
            d1 = 0, d2 = 80, x1 = 0, x2 = 160, y1 = 0, y2 = 128
            """
            image_size = image.shape
            overlap = 1 / 3
            classes = 2

            strideHW = ceil(input_size[1] * (1 - overlap))
            strideD = ceil(input_size[0] * (1 - overlap))
            tile_deps = int(ceil((image_size[2] - input_size[0]) / strideD) + 1)
            tile_rows = int(ceil((image_size[3] - input_size[1]) / strideHW) + 1)
            tile_cols = int(ceil((image_size[4] - input_size[2]) / strideHW) + 1)
            pred = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(
                np.float32)
            count_predictions = np.zeros(
                (image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
            pred = torch.from_numpy(pred).cuda()
            count_predictions = torch.from_numpy(count_predictions).cuda()
            # print(f'strideHW = {strideHW}, strideD = {strideD}')
            # print(f'tile_deps = {tile_deps}, tile_rows = {tile_rows}, tile_cols = {tile_cols}')

            for dep in range(tile_deps):
                for row in range(tile_rows):
                    for col in range(tile_cols):
                        d1 = int(dep * strideD)
                        x1 = int(col * strideHW)
                        y1 = int(row * strideHW)
                        d2 = min(d1 + input_size[0], image_size[2])
                        x2 = min(x1 + input_size[2], image_size[4])
                        y2 = min(y1 + input_size[1], image_size[3])
                        d1 = max(int(d2 - input_size[0]), 0)
                        x1 = max(int(x2 - input_size[2]), 0)
                        y1 = max(int(y2 - input_size[1]), 0)
                        # print(f'd1 = {d1}, d2 = {d2}, x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}')

                        img = image[:, :, d1:d2, y1:y2, x1:x2]
                        img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                        prediction = model([img, img_res])
                        prediction = prediction[0]

                        count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                        pred[:, :, d1:d2, y1:y2, x1:x2] += prediction
                        # print(f'count_predictions = {count_predictions.size()}, pred = {pred}')

            pred /= count_predictions
            pred = F.sigmoid(pred)
            dice = utils.metrics.compute_channel_dice(pred, label)
            # Area under the ROC curve
            # from sklearn.metrics import roc_curve
            # fpr, tpr, thresholds = roc_curve((label), pred)
            # AUC_ROC = roc_auc_score(label, pred)
            # print(f'Area under the ROC curve = {str(AUC_ROC)}')
            # roc_curve = plt.figure()
            # plt.plot(fpr, tpr, '-',label=f'Area Under the Curve (AUC = {AUC_ROC:.4f}')
            # plt.title(f'ROC Curve')
            # plt.xlabel('FPR (False Positive Rate)')
            # plt.ylabel('TPR (True Positive Rate)')
            # plt.legend(loc='lower right')
            # plt.savefig(PATH+'ROC.png')

            primary_nonzero = label[:, 0, :, :, :].nonzero()
            lymph_nonzero = label[:, 1, :, :, :].nonzero()

            if primary_nonzero.nelement() == 0:
                print(f'No primary tumor')
                primary_dice.append(float('NaN'))
                lymph_dice.append(dice[0])
                print(f'DSC: {dice[0]:.4f}     Primary: None     Lymph: {dice[0]:.4f}')
            elif lymph_nonzero.nelement() == 0:
                print(f'No lymph node')
                primary_dice.append(dice[0])
                lymph_dice.append(float('NaN'))
                print(f'DSC: {dice[0]:.4f}     Primary: {dice[0]:.4f}     Lymph: None')
            else:
                # both primary and lymph exist
                primary_dice.append(dice[0])
                lymph_dice.append(dice[1])
                print(f'DSC: {dice.mean():.4f}     Primary: {dice[0]:.4f}     Lymph: {dice[1]:.4f}')

            # val += dice
            # print(f'DSC: {dice.mean():.4f}     Primary: {dice[0]:.4f}     Lymph: {dice[1]:.4f}')
            # print(f'pred size = {pred.size()}, label = {label.size()}')
            # pred size = torch.Size([1, 2, 80, 128, 160]), label = torch.Size([1, 2, 80, 128, 160])

            # hausdorff_dist = utils.metrics.compute_channel_hausdorff(pred, label)
            # hausdorff_dist = utils.metrics.hausdorff_distance(pred, label)
            # print(f'hausdorff distance = {hausdorff_dist}')

            pred_arr = pred.cpu().numpy()
            label_arr = label.cpu().numpy()
            pred_arr = np.where(pred_arr > 0.5, 1, 0)

            primary_matrix = ConfusionMatrix(pred=pred_arr[:, 0, :, :, :], label=label_arr[:, 0, :, :, :])
            lymph_matrix = ConfusionMatrix(pred=pred_arr[:, 1, :, :, :], label=label_arr[:, 1, :, :, :])

            tp, fp, tn, fn = primary_matrix.get_matrix()
            print(f'Primary: tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}, Total = {tp + fp + tn + fn}')
            print(f'Total primary tumor area = {tp + fn}')
            tp, fp, tn, fn = lymph_matrix.get_matrix()
            print(f'Lymph: tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}, Total = {tp + fp + tn + fn}')
            print(f'Total lymph node area = {tp + fn}')

            recall_p = utils.metrics.recall(confusion_matrix=primary_matrix)
            recall_l = utils.metrics.recall(confusion_matrix=lymph_matrix)
            print(f'recall p = {recall_p:.4f}       recall l = {recall_l:.4f}')

            precision_p = utils.metrics.precision(confusion_matrix=primary_matrix)
            precision_l = utils.metrics.precision(confusion_matrix=lymph_matrix)
            print(f'precision p = {precision_p:.4f}     precision l = {precision_l:.4f}')

            fscore_p = utils.metrics.f1_score(confusion_matrix=primary_matrix)
            fscore_l = utils.metrics.f1_score(confusion_matrix=lymph_matrix)
            print(f'f1 score p = {fscore_p:.4f}     f1 score l = {fscore_l:.4f}')

            hausdorff_distance_p = utils.metrics.hausdorff_distance(confusion_matrix=primary_matrix)
            hausdorff_distance_l = utils.metrics.hausdorff_distance(confusion_matrix=lymph_matrix)
            print(f'hausdorff_distance p = {hausdorff_distance_p:.4f}       hausdorff_distance l = {hausdorff_distance_l:.4f}')

            hausdorff_distance_95_p = utils.metrics.hausdorff_distance_95(confusion_matrix=primary_matrix)
            hausdorff_distance_95_l = utils.metrics.hausdorff_distance_95(confusion_matrix=lymph_matrix)
            print(f'hausdorff_distance 95 p = {hausdorff_distance_95_p:.4f}       hausdorff_distance 95 l = {hausdorff_distance_95_l:.4f}')
            eval_metrics = {}
            eval_metrics.update({'dice_p': primary_dice[index], 'dice_l': lymph_dice[index],
                                 'recall_p': recall_p, 'recall_l': recall_l,
                                 'precision_p': precision_p, 'precision_l': precision_l,
                                 'fscore_p': fscore_p, 'fscore_l': fscore_l,
                                 'hausdorff_distance_p': hausdorff_distance_p, 'hausdorff_distance_l': hausdorff_distance_l,
                                 'hausdorff_distance_95_p': hausdorff_distance_95_p,
                                 'hausdorff_distance_95_l': hausdorff_distance_95_l})
            eval_list.append(eval_metrics)

            # print(f'eval list = {eval_list}, len = {len(eval_list)}')
            # evaluate = pd.DataFrame(columns=['recall_p', 'recall_l'])
            # evaluate.loc[index] = recall_p, recall_l
            # f1_primary, f1_lymph = utils.metrics.compute_f1_score(pred, label)
            # f1_p_list.append(f1_primary)
            # f1_l_list.append(f1_lymph)
            # pred = F.sigmoid(pred)

            pred = pred.squeeze()
            pred_arr = pred.cpu().numpy()
            # print(f'pred min = {np.min(pred_arr)}, max = {np.max(pred_arr)}')

            patient_num = path_list[index].split('\\')[-2]
            file_name = f'ConResNet_Pred_{patient_num}.nii.gz'

            pred_arr = np.where(pred_arr > 0.5, 1, 0)
            # create combined array of left and right labels
            pred_combined_arr = pred_arr[0, :, :, :] + pred_arr[1, :, :, :]
            # output_arr = np.where(output_arr > 0, 1, 0)

            # if value is 2, change to 1
            pred_combined_arr = np.where(pred_combined_arr > 1, 1, pred_combined_arr)
            # print(f'pred combine min = {np.min(pred_combined_arr)}, max = {np.max(pred_combined_arr)}')
            pred_combined = sitk.GetImageFromArray(pred_combined_arr)
            os.chdir(path_list[index])
            # sitk.WriteImage(pred_combined[:, :, :], file_name)

            print(f'{file_name} saved in {os.getcwd()}\n')
        # break

    print(f'Evaluation dataframe: ')
    eval_df = pd.DataFrame(eval_list, columns=['dice_p', 'dice_l', 'recall_p', 'recall_l', 'precision_p', 'precision_l',
                                               'fscore_p', 'fscore_l',
                                                'hausdorff_distance_p', 'hausdorff_distance_l',
                                                'hausdorff_distance_95_p', 'hausdorff_distance_95_l'])
    # Add row of Mean value of each metrics
    eval_df.loc['Mean'] = eval_df.mean()
    eval_df.loc['Median'] = eval_df.median()
    eval_df.loc['Std'] = eval_df.std()
    print(eval_df)
    eval_df.to_csv(PATH + '/prediction_valid_BEST.csv', mode='w')
    print(f'Evaluation csv saved in {os.getcwd()}')

    print('End of validation')
    print(f'len(primary_dice) = {len(primary_dice)}, len(lymph_dice) = {len(lymph_dice)}')
    print(
        f'Total DSC: {(sum(primary_dice) + sum(lymph_dice)) / (len(primary_dice) + len(lymph_dice)):.4f}      '
        f'Primary: {sum(primary_dice) / len(primary_dice):.4f}     Lymph: {sum(lymph_dice) / len(lymph_dice):.4f}')


_, _, pred_loader = dataset.lung_dataloader.generate_lung_dataset()
PATH = 'F:/ContextLearning/snapshots/ConResNet_1001_1113'

predictor(PATH=PATH, data_loader=pred_loader)