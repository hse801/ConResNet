import argparse
import sys
sys.path.append("..")

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
# from models.ConResNet_no_residual import ConResNet_no_res
from models.ConResNet_Co import ConResNet_Co
# from models.ConResNet_sobel import ConResNet_sobel
# from models.ConResNet_2_slice import ConResNet_2_slice
from models.ConResNet_new import COCONNET
from models.ConResNet_mod import ConResNet_mod
from models.Co_Con_Aspp import Co_Con_ASPP
import timeit
from tensorboardX import SummaryWriter
from utils import loss
from utils.engine import Engine
from math import ceil
import utils.metrics
import matplotlib.pyplot as plt
import pandas as pd


from dataset.thyroid_dataloader import generate_thyroid_dataset
start = timeit.default_timer()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ConResNet for 3D Medical Image Segmentation.")

    # parser.add_argument("--data_dir", type=str, default='/media/userdisk0/myproject-Seg/BraTS-pro/dataset/')
    # parser.add_argument("--train_list", type=str, default='list/BraTS2018_old/train.txt')
    # parser.add_argument("--val_list", type=str, default='list/BraTS2018_old/val.txt')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/')
    parser.add_argument("--reload_path", type=str, default='snapshots/conresnet/ConResNet_40000.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='80, 128, 160')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=1234)

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def validate(input_size, model, ValLoader, num_classes=1):
    # start to validate
    val = 0.0

    for index, batch in enumerate(ValLoader):
        # print('%d processd'%(index))
        # image, image_res, label, size, name, affine = batch
        image, image_res, label = batch
        image = image.cuda()
        image_res = image_res.cuda()
        label = label.cuda()
        with torch.no_grad():
            pred = utils.metrics.predict_sliding(model, [image, image_res], input_size, num_classes)
            # print('predict sliding done')
            # print(f'pred mean = {pred.mean()}, label = {label.mean()}')
            pred = F.sigmoid(pred)
            dice = utils.metrics.compute_channel_dice(pred, label)
            # print(f'dice = {dice}')
            # print('compute dice score done')
            val += dice
            print(f'dice = {dice}, val = {val}')

    return val / (index+1)


def main():
    """Create the ConResNet model and then start the training."""
    parser = get_arguments()
    print(parser)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)
        now = time.localtime()
        # os.mkdir(args.snapshot_dir + f'{now.tm_mon}{now.tm_mday}_{now.tm_hour}{now.tm_min}', exist_ok=True)
        log_dir = args.snapshot_dir + 'ConResNet_{:02}{:02}_{:02}{:02}/'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
        os.makedirs(log_dir, exist_ok=True)
        print(f'{log_dir} folder created')

        writer = SummaryWriter(log_dir)

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # print(f'input size = {input_size}')
        model = Co_Con_ASPP(input_size, num_classes=args.num_classes, weight_std=True)
        # model = ConResNet_mod(input_size, num_classes=args.num_classes, weight_std=True)
        # model = ConResNet_Co(input_size, num_classes=args.num_classes, weight_std=True)
        print(model)
        # model.train()
        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        optimizer = optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
            lr=args.learning_rate, weight_decay=args.weight_decay)
        # learning rate scheduler
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        loss_D = loss.DiceLoss().to(device)
        loss_BCE = loss.BCELoss().to(device)

        loss_B = loss.BCELossBoud().to(device)

        # if not os.path.exists(args.snapshot_dir):
        #     os.makedirs(args.snapshot_dir)

        # trainloader, valloader, _ = generate_thyroid_dataset()

        trainloader = engine.get_train_loader()
        valloader = engine.get_test_loader()
        # trainloader, train_sampler = engine.get_train_loader(BraTSDataSet(args.data_dir, args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
        #                 scale=args.random_scale, mirror=args.random_mirror))
        # valloader, val_sampler = engine.get_test_loader(BraTSValDataSet(args.data_dir, args.val_list))
        best_dice = 0
        min_loss = 100
        lrs = []
        metrics_list = []
        for epoch in range(args.epochs):
            total_loss_sum = 0
            for i_iter, batch in enumerate(trainloader):
                i_iter += args.start_iters
                images, images_res, labels, labels_res = batch
                # print(f'train_conresnet: images = {images.size()}, images_res = {images_res.size()}, labels = {labels.size()}, res = {labels_res.size()}')
                # print(f'args.local_rank = {args.local_rank}')
                # print(f'images == images_res = {images == images_res}')
                # print(f'labels == labels_res = {np.array_equal(labels, labels_res)}')
                images = images.cuda()
                images_res = images_res.cuda()
                labels = labels.cuda()
                labels_res = labels_res.cuda()

                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

                preds = model([images, images_res])
                preds_seg = preds[0]
                preds_res = preds[1]
                preds_resx2 = preds[2]
                preds_resx4 = preds[3]
                # print(f'train_conresnet: preds_seg = {preds_seg.size()}, preds_res = {preds_res.size()}, preds_resx2 = {preds_resx2.size()}, preds_resx4 = {preds_resx4.size()}')

                # preds_seg = torch.Size([1, 1, 80, 128, 160]), preds_res = torch.Size([1, 1, 80, 128, 160]),
                # preds_resx2 = torch.Size([1, 1, 80, 128, 160]), preds_resx4 = torch.Size([1, 1, 80, 128, 160])

                term_seg_Dice = loss_D.forward(preds_seg, labels)
                term_seg_BCE = loss_BCE.forward(preds_seg, labels)

                term_res_BCE = loss_B.forward(preds_res, labels_res)
                term_resx2_BCE = loss_B.forward(preds_resx2, labels_res)
                term_resx4_BCE = loss_B.forward(preds_resx4, labels_res)

                term_all = term_seg_Dice + term_seg_BCE + term_res_BCE + 0.5 * (term_resx2_BCE + term_resx4_BCE)
                total_loss_sum += term_all
                term_all.backward()

                optimizer.step()

                # if i_iter % 100 == 0 and (args.local_rank == 0):
                writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar('loss', term_all.cpu().data.numpy(), epoch)
                if i_iter % 20 == 0:
                    print('Epoch = {}, iter = {} : seg_loss = {:.4}, res_loss = {:.4}, total_loss = {:.4}'.format(
                            epoch, i_iter, (term_seg_Dice+term_seg_BCE).cpu().data.numpy(),
                            (term_res_BCE+term_resx2_BCE+term_resx4_BCE).cpu().data.numpy(),
                            term_all.cpu().data.numpy()))

                # if i_iter >= args.num_steps - 1 and (args.local_rank == 0):
                #     print('save last model ...')
                #     torch.save(model.state_dict(), osp.join(log_dir, 'ConResNet_' + str(args.num_steps) + '.pth'))
                #     break
                #
                # if i_iter % args.val_pred_every == 0 and i_iter != 0 and (args.local_rank == 0):
                #     print('save model ...')
                #     torch.save(model.state_dict(), osp.join(log_dir, 'ConResNet_' + str(i_iter) + '.pth'))

                # val
            # if i_iter % args.val_pred_every == 0:
            # scheduler.step()
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
            lrs.append(optimizer.param_groups[0]["lr"])
            print('Validate ...')
            val = validate(input_size, model, valloader, args.num_classes)
            total_loss_sum /= len(trainloader)
            # print(f'val type = {type(val)}, val = {val}')
            # if (args.local_rank == 0):

            writer.add_scalar('Val_Dice', val[0], val[1], epoch)
            print('Validate Epoch: {}, Primary: {:.4}, Lymph: {:.4}, Mean dice: {:.4}'.format(epoch, val[0], val[1], val.mean()))
            print(f'Total loss of epoch {epoch}: {total_loss_sum}')

            print(f'val.mean() = {val.mean():.4f}, best_dice = {best_dice:.4f}')
            if val.mean() > best_dice:
                print(f'Save best model at epoch {epoch}...')
                best_dice = val.mean()
                torch.save(model.state_dict(), osp.join(log_dir, 'ConResNet_BEST' + '.pth'))
            if total_loss_sum < min_loss:
                print(f'Save model with minimum loss at epoch {epoch} ...')
                torch.save(model.state_dict(), osp.join(log_dir, f'ConResNet_min_loss' + '.pth'))
            if epoch % 20 == 0:
                print(f'Save model every 20 epoch at epoch {epoch}...')
                torch.save(model.state_dict(), osp.join(log_dir, f'ConResNet_Epoch{epoch}' + '.pth'))

            print(f'Save last model at epoch {epoch}...')
            torch.save(model.state_dict(), osp.join(log_dir, 'ConResNet_LAST' + str(epoch) + '.pth'))

            train_metrics = {}
            train_metrics.update({'Dice_p': val[0], 'Dice_l': val[1], 'Dice_mean': val.mean(), 'Loss': total_loss_sum})
            metrics_list.append(train_metrics)
        end = timeit.default_timer()
        metric_df = pd.DataFrame(metrics_list, columns=['Dice_p', 'Dice_l', 'Dice_mean', 'Loss'])
        metric_df.to_csv(log_dir + '/train_metrics.csv', mode='w')
        print(f'Total training time: {(end - start) / 60} minutes')

    end = timeit.default_timer()

    print(f'Total training time: {(end - start) / 60} minutes')


if __name__ == '__main__':
    main()



