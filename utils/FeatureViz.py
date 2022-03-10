import torch
import numpy as np
import os, argparse, cv2
from lib.Network_Res2Net_GRA_NCD_FeatureViz import Network
from utils.dataloader import test_dataset


def heatmap(feat_viz, ori_img, save_path=None):
    feat_viz = torch.mean(feat_viz, dim=1, keepdim=True).data.cpu().numpy().squeeze()
    feat_viz = (feat_viz - feat_viz.min()) / (feat_viz.max() - feat_viz.min() + 1e-8)

    ori_img = ori_img.data.cpu().numpy().squeeze()
    ori_img = ori_img.transpose((1, 2, 0))
    ori_img = ori_img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    ori_img = ori_img[:, :, ::-1]
    # img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ori_img = np.uint8(255 * ori_img)
    feat_viz = np.uint8(255 * feat_viz)
    feat_viz = cv2.applyColorMap(feat_viz, cv2.COLORMAP_JET)
    feat_viz = cv2.resize(feat_viz, (320, 320))
    ori_img = cv2.resize(ori_img, (320, 320))
    # print(feat_viz.shape, ori_img.shape)
    feat_viz = cv2.addWeighted(ori_img, 0.5, feat_viz, 0.5, 0)

    cv2.imwrite(save_path, feat_viz)