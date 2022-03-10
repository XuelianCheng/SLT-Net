from __future__ import print_function, division
import sys
# sys.path.append('lib')
sys.path.append('dataloaders')

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import pdb
from datetime import datetime
from torchvision.utils import make_grid

from lib import VideoModel_pvtv2 as Network
from dataloaders import video_dataloader
from utils.pyt_utils import load_model
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, data_blob in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = [x.cuda() for x in data_blob[0]]
            gts = data_blob[1].cuda()
            preds1, preds2, pred = model(images)

            loss1 = structure_loss(preds1[0], gts) + structure_loss(preds1[1], gts) + structure_loss(preds1[2], gts) 
            loss2 = structure_loss(preds2[0], gts) + structure_loss(preds2[1], gts) + structure_loss(preds2[2], gts) 
            loss_final = structure_loss(pred, gts) + structure_loss(preds1[3], gts) + structure_loss(preds2[3], gts)

            loss = loss1 + loss2 + loss_final

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 1 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss_final: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss1.data, loss2.data,  loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f}  Loss2: {:0.4f} Loss_final: {:0.4f}'.
                     format(epoch, opt.epoch, i, total_step, loss.data, loss1.data, loss2.data,  loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss1': loss1.data, 'Loss2': loss2.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds1[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_01', torch.tensor(res), step, dataformats='HW')
                res = preds2[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_02', torch.tensor(res), step, dataformats='HW')
                res = pred[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        #if epoch % 50 == 0:
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            images, gt, name, scene = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            images = [x.cuda() for x in images]

            _,_,res = model(images)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

def freeze_network(model):
    for name, p in model.named_parameters():
        if "fusion_conv" not in name:
            p.requires_grad = False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--trainsplit', type=str, default='MoCA-Video-Train', help='train from checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--pretrained_cod10k', default=None,
                        help='path to the pretrained Swin Transformer')

    parser.add_argument('--resume', type=str, default='', help='train from checkpoints')
    parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=123')
    parser.add_argument('--dataset',  type=str, default='MoCA')
    parser.add_argument('--save_path', type=str,default='./snapshot/MoCA/',
                        help='the path to save model and log')
    parser.add_argument('--valonly', action='store_true', default=False, help='skip training during training')
    opt = parser.parse_args()

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # build the model
    model = Network(opt).cuda()
    # pdb.set_trace()

    if os.path.exists(opt.resume):
        model = load_model(model=model, model_file=opt.resume, is_restore=True)
        print('Loading state dict from: {0}'.format(opt.resume))
    else:
        print("Cannot find model file at {}".format(opt.resume))

    #freeze_network(model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('===> Loading datasets')

    train_loader, val_loader = video_dataloader(opt)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.resume, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        if not opt.valonly:
            train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)
