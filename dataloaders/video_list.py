import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import time
from glob import glob
import os.path as osp
import pdb
from mypath import Path

# several data augumentation strategies
def cv_random_flip(imgs, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, label

def randomCrop(imgs, label):
    border = 30
    image_width = imgs[0].size[0]
    image_height = imgs[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    
    for i in range(len(imgs)):
        imgs[i] = imgs[i].crop(random_region)
    return imgs, label.crop(random_region)

def randomRotation(imgs, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        for i in range(len(imgs)):
            imgs[i] = imgs[i].rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return imgs, label

def colorEnhance(imgs):
    for i in range(len(imgs)):
        bright_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Brightness(imgs[i]).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        imgs[i] = ImageEnhance.Contrast(imgs[i]).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        imgs[i] = ImageEnhance.Color(imgs[i]).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        imgs[i] = ImageEnhance.Sharpness(imgs[i]).enhance(sharp_intensity)
    return imgs

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255       
    return Image.fromarray(img)

class VideoDataset(data.Dataset):
    def __init__(self, dataset='MoCA', trainsize=256, split='MoCA-Video-Train'):
        self.trainsize = trainsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []

        if dataset == 'MoCA': 
            root = Path.db_root_dir('MoCA')
            img_format = '*.jpg'
            data_root = osp.join(root, split)

            for scene in os.listdir(osp.join(data_root)):
                if split=='MoCA-Video-Train':
                    images  = sorted(glob(osp.join(data_root, scene, 'Frame', img_format)))
                elif split=='TrainDataset_per_sq':
                    images  = sorted(glob(osp.join(data_root, scene, 'Imgs', img_format)))
                gt_list = sorted(glob(osp.join(data_root, scene, 'GT', '*.png')))
                # pdb.set_trace()

                for i in range(len(images)-2):
                    self.extra_info += [ (scene, i) ]  # scene and frame_id
                    self.gt_list    += [ gt_list[i] ]
                    self.image_list += [ [images[i], 
                                       images[i+1], 
                                       images[i+2]] ]

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        imgs = []
        names= []
        index = index % len(self.image_list)

        for i in range(len(self.image_list[index])):
            imgs += [self.rgb_loader(self.image_list[index][i])]
            names+= [self.image_list[index][i].split('/')[-1]]

        #print(names)
        scene= self.image_list[index][0].split('/')[-3]  
        gt = self.binary_loader(self.gt_list[index])

        imgs, gt = cv_random_flip(imgs, gt)
        #imgs, gt = randomCrop(imgs, gt)
        imgs, gt = randomRotation(imgs, gt)
        imgs = colorEnhance(imgs)
        gt = randomPeper(gt)

        for i in range(len(imgs)):
            imgs[i] = self.img_transform(imgs[i])
        gt = self.gt_transform(gt)

        return imgs, gt#, scene, names

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L') 

    def __len__(self):
        return len(self.image_list)

# dataloader for training
def get_loader(dataset, batchsize, trainsize, train_split,
    shuffle=True, num_workers=12, pin_memory=True):
    dataset = VideoDataset(dataset, trainsize, split=train_split)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, dataset='MoCA', split='TestDataset_per_sq', testsize=256):
        self.testsize = testsize
        self.image_list = []
        self.gt_list = []
        self.extra_info = []

        if dataset == 'CAD2016':    
            root = Path.db_root_dir('CAD2016')
            img_format = '*.png'

            for scene in os.listdir(osp.join(root)):
                images  = sorted(glob(osp.join(root, scene, 'frames', img_format)))
                gt_list = sorted(glob(osp.join(root, scene, 'pseudo', '*.png')))

                for i in range(len(images)-2):
                    self.extra_info += [ (scene, i) ]  # scene and frame_id
                    self.gt_list    += [ gt_list[i] ]
                    self.image_list += [ [images[i], 
                                       images[i+1], 
                                       images[i+2]] ]


                for i in range(len(images)-1, len(images)-3,-1): 
                    self.gt_list    += [ gt_list[i] ]
                    self.image_list += [ [images[i], 
                                    images[i-1], 
                                    images[i-2]] ]

        else: 
            root = Path.db_root_dir('MoCA')
            img_format = '*.jpg'
            data_root = osp.join(root, split)
            print(split)

            for scene in os.listdir(osp.join(data_root)):
                if split=='MoCA-Video-Test':
                    images  = sorted(glob(osp.join(data_root, scene, 'Frame', img_format)))
                elif split=='TestDataset_per_sq':
                    images  = sorted(glob(osp.join(data_root, scene, 'Imgs', img_format)))
                gt_list = sorted(glob(osp.join(data_root, scene, 'GT', '*.png')))

                for i in range(len(images)-2):
                    self.extra_info += [ (scene, i) ]  # scene and frame_id
                    self.gt_list    += [ gt_list[i] ]
                    self.image_list += [ [images[i], 
                                       images[i+1], 
                                       images[i+2]] ]


        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        self.index = 0
        self.size = len(self.gt_list)

    def load_data(self):
        imgs = []
        names= []

        for i in range(len(self.image_list[self.index])):
            imgs += [self.rgb_loader(self.image_list[self.index][i])]
            names+= [self.image_list[self.index][i].split('/')[-1]]
            imgs[i] = self.transform(imgs[i]).unsqueeze(0)

        scene= self.image_list[self.index][0].split('/')[-3]  
        gt = self.binary_loader(self.gt_list[self.index])

        self.index += 1
        self.index = self.index % self.size
    
        return imgs, gt, names, scene

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
        