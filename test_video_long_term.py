import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os, argparse
from PIL import Image
from lib import VideoModel_long_term as Network
from mypath import Path
from glob import glob
import os.path as osp
import imageio
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--frame_gap', type=int, default=1, help='epoch number')
parser.add_argument('--input_length', type=int, default=5, help='epoch number')
parser.add_argument('--fsampling_rate', type=int, default=1, help='epoch number')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--dataset',  type=str, default='MoCA')
parser.add_argument('--testsplit',  type=str, default='MoCA-Video-Test')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/COD10K/Net_epoch_best.pth')
parser.add_argument('--short_pretrained', type=str, default=None, help='train from short_term_architure')
opt = parser.parse_args()

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6

class test_dataset:
    def __init__(self, dataset='MoCA', split='MoCA-Video-Test',
                input_length=10, fsampling_rate=1):
        self.input_length = input_length
        self.fsampling_rate = fsampling_rate
        self.image_list = []
        self.extra_info = []

        if dataset == 'CAD2016':    
            root = Path.db_root_dir('CAD2016')
            img_format = '*.png'
        elif dataset == 'MoCA': 
            root = Path.db_root_dir('MoCA')
            img_format = '*.jpg'

        data_root = osp.join(root, split)

        for scene in os.listdir(osp.join(data_root)):
            images  = sorted(glob(osp.join(data_root, scene, 'Pred', '*.png')))

            clip_size = self.input_length
            skip_size = self.input_length
            out = False

            video_len = len(images)
            for i in range(1, video_len, skip_size):
                clip_im = []
                indices = list(range(i, min(i+clip_size*(self.fsampling_rate), video_len), self.fsampling_rate))
                if len(indices) < clip_size:
                    continue
                for j in indices:
                    clip_im.append(images[j])
                self.image_list.append(clip_im)

            # add last one    
            for i in range(video_len-1, video_len-self.input_length-2, -fsampling_rate): 

                clip_im = []
                indices = list(range(i, min(i+clip_size*(self.fsampling_rate), video_len), self.fsampling_rate))
                if len(indices) < clip_size:
                    continue
                for j in indices:
                    clip_im.append(images[j])
                self.image_list.append(clip_im)

            # add first one    
            clip_im = []
            indices=list(range(self.input_length, -1, -self.fsampling_rate))
            for j in indices:
                clip_im.append(images[j])
            self.image_list.append(clip_im)


        if len(self.image_list) == 0:
            raise
        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 448)),
            transforms.ToTensor()])

        self.index = 0
        self.size = len(self.image_list)

    def load_data(self):
        imgs = []
        shts = []
        names= []
        IMG = None
        PRED= None
        LABEL = None
        # forward
        for i in range(len(self.image_list[self.index])):
        # backward
        # for i in range(len(self.image_list[self.index])-1, -1, -1): 
            if 'MoCA-Video-Test' in self.image_list[self.index][i]:
                rgb_name = self.image_list[self.index][i].replace('Pred','Frame')
            else:
                rgb_name = self.image_list[self.index][i].replace('Pred','Imgs')
            rgb_name = rgb_name.replace('.png','.jpg')
            imgs += [self.rgb_loader(rgb_name)]
            shts += [self.binary_loader(self.image_list[self.index][i])]
            names+= [self.image_list[self.index][i].split('/')[-1]]

        img_size = imgs[0].size

        for i in range(len(imgs)):
            imgs[i] = self.transform(imgs[i]).unsqueeze(0)
            shts[i] = self.transform(shts[i]).unsqueeze(0)

        scene= self.image_list[self.index][0].split('/')[-3] 
        self.index += 1
        self.index = self.index % self.size

        for idx, (img, sht) in enumerate(zip(imgs, shts)):
            if IMG is not None:
                IMG[idx, :, :, :] = img
                PRED[idx, :, :, :] = sht
            else:
                IMG = torch.zeros(len(imgs), *(img.shape))
                PRED = torch.zeros(len(imgs), *(sht.shape))
                IMG[idx, :, :, :] = img
                PRED[idx, :, :, :] = sht
        return IMG, PRED, img_size, names, scene

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size

def test_dataloader(args):   
    test_loader = test_dataset(dataset=args.dataset,
                              split=args.testsplit,
                              input_length=args.input_length,
                              fsampling_rate=args.fsampling_rate)
    print('Test with %d image pairs' % len(test_loader))
    return test_loader 

if __name__ == '__main__':
    test_loader = test_dataloader(opt)
    name_e = opt.pth_path.split('/')[-1].split('_')[-1]
    save_root = './res/{}/longterm_{}_f{}/'.format(opt.dataset, name_e[:-4], opt.input_length)
    # pdb.set_trace()
    model = Network(opt)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # compute parameters
    print('Total Params = %.2fMB' % count_parameters_in_MB(model))

    for i in range(test_loader.size):
        images, shorts, gt_shape, names, scene = test_loader.load_data()
        save_path=save_root+scene+'/Pred/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        
        inputs = torch.cat([images, shorts], 2)

        preds = model(inputs)

        for res, name in zip(preds[-1][:], names[:]):
            if name[-5] in ['0','5']:
                # pdb.set_trace()
                res = F.upsample(res.unsqueeze(0), size=(gt_shape[1],gt_shape[0]), mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                print('> ')
                # name =names[index].replace('jpg','png')
                imageio.imwrite(save_path+name, res)