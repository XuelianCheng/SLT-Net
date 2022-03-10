import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
from lib import VideoModel_pvtv2 as Network
from dataloaders import test_dataloader
import imageio
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='MoCA')
parser.add_argument('--testsplit',  type=str, default='MoCA-Video-Test')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/COD10K/Net_epoch_best.pth')
parser.add_argument('--pretrained_cod10k', default=None,
                        help='path to the pretrained Resnet')
opt = parser.parse_args()

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6

if __name__ == '__main__':
    test_loader = test_dataloader(opt)
    save_root = './res/{}/'.format(opt.dataset)
    # pdb.set_trace()
    model = Network(opt)

    # pretrained_dict = torch.load(opt.pth_path)
    # model_dict = model.state_dict()
    # #pdb.set_trace()
    # for k, v in pretrained_dict.items():
    #     pdb.set_trace()

    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # compute parameters
    print('Total Params = %.2fMB' % count_parameters_in_MB(model))

    for i in range(test_loader.size):
        images, gt, names, scene = test_loader.load_data()
        save_path=save_root+scene+'/Pred/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        images = [x.cuda() for x in images]

        res1, res2, res = model(images)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = F.upsample(res1[-1], size=gt.shape, mode='bilinear', align_corners=False)
        # res = F.upsample(res2[-1], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> ')

        name =names[0].replace('jpg','png')

        # if name[-5] in ['0','5']:
        imageio.imwrite(save_path+name, res)
