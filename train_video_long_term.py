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

from lib import VideoModel_long_term as Network
from dataloaders import video_dataloader_long
from utils.pyt_utils import load_model
from utils.utils import clip_gradient, adjust_lr
from utils.cyclic_scheduler import CyclicLRWithRestarts
from utils.adamw import AdamW
from utils.Hybrid_Eloss import hybrid_e_loss
from tensorboardX import SummaryWriter


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, scheduler, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all  = 0
    epoch_step = 0
    try:
        
        for i, data_blob in enumerate(train_loader, start=1):
            scheduler.step()
            optimizer.zero_grad()

            images = data_blob[0].cuda()
            shorts = data_blob[1].cuda()
            gts = data_blob[2].cuda()

            inputs = torch.cat([images, shorts], 2)
            preds = model(inputs)

            gts_sup = gts.view(-1, *(gts.shape[2:]))

            # loss = structure_loss(preds[0], gts_sup) + structure_loss(preds[1], gts_sup) + structure_loss(preds[2], gts_sup) + structure_loss(preds[3], gts_sup)
            loss = hybrid_e_loss(preds[0], gts_sup) + hybrid_e_loss(preds[1], gts_sup) + hybrid_e_loss(preds[2], gts_sup) + hybrid_e_loss(preds[3], gts_sup)
            
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            scheduler.batch_step()

            step += 1
            epoch_step += 1
            loss_all += loss.mean().data

            if i % 1 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_total': loss.data},
                                   global_step=step)

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
            images,shorts, gt, name, scene = test_loader.load_data()

            inputs = torch.cat([images, shorts], 2)
            preds = model(inputs)

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            images = [x.cuda() for x in images]

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            #pdb.set_trace()
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
            #print('freeze layer: {}'.format(name))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--input_length', type=int, default=5, help='epoch number')
    parser.add_argument('--fsampling_rate', type=int, default=1, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--beta', type=float, default=0.0005, help='weighting on KL')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')

    parser.add_argument('--resume', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--short_pretrained', type=str, default=None, help='train from short_term_architure')
    parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--seed', type=int, default=2021, help='random seed to use. Default=123')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--dataset',  type=str, default='MoCA')
    parser.add_argument('--save_path', type=str,default='./snapshot/MoCA/',
                        help='the path to save model and log')
    parser.add_argument('--valonly', action='store_true', default=False, help='skip training during training')
    opt = parser.parse_args()

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.backends.cudnn.benchmark = True
    # build the model
    model = Network(opt)
    model = torch.nn.DataParallel(model).cuda()

    if opt.resume is not None:
        model.load_state_dict(torch.load(opt.resume), strict=True)
        print('Loading model from ', opt.resume)


    #freeze_network(model)
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    optimizer = AdamW(model.parameters(), opt.lr, weight_decay=1e-6)
    scheduler = CyclicLRWithRestarts(optimizer, opt.batchsize, epoch_size=1024, restart_period=5, t_mult=1.2, policy="cosine")

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('===> Loading datasets')

    train_loader, val_loader =  video_dataloader_long(opt)
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
        #cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', opt.lr, global_step=epoch)
        if not opt.valonly:
            train(train_loader, model, optimizer, scheduler, epoch, save_path, writer)
        #val(val_loader, model, epoch, save_path, writer)
