import torch
import torch.nn as nn
import pdb

from lib.pvtv2_afterTEM import Network
from lib.ref_video.PNS_Module import NS_Block

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class NonLocalNet(nn.Module):
    def __init__(self, in_planes, pyramid_type='conv'):
        super(NonLocalNet, self).__init__()
        self.pyramid_type = pyramid_type

        self.NSB_1 = NS_Block(in_planes)
        self.NSB_2 = NS_Block(in_planes)

    def forward(self, fea, origin_shape):
        fea1 = fea[1].view([*origin_shape[:2], *fea[1].shape[1:]])
        high_feature_1 = self.NSB_1(fea1) + fea1
        high_feature_2 = self.NSB_2(high_feature_1) + high_feature_1
        out2 = fea1 + high_feature_2
        out2 = out2.view(-1, *out2.shape[2:]) 
        return fea[0], out2, fea[2]

class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.args = args
        self.backbone = Network(pvtv2_pretrained=False, imgsize=self.args.trainsize)
    def forward(self, frame):
        seg = self.backbone(frame)
        return seg

class VideoModel(nn.Module):
    def __init__(self, args):
        super(VideoModel, self).__init__()

        self.args = args
        self.extra_channels = 0
        print("Select mask mode: concat, num_mask={}".format(self.extra_channels))

        self.backbone = Network(pvtv2_pretrained=False, imgsize=self.args.trainsize)
        if self.args.short_pretrained is not None:
            self.load_backbone(self.args.short_pretrained )

        self.nlnet = NonLocalNet(in_planes=32, pyramid_type='conv')
        self.first_conv = nn.Conv2d(4, 3, 1)

        self.freeze_bn()

    def load_backbone(self, pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = self.state_dict()
        print("Load pretrained parameters from {}".format(pretrained))
        # pdb.set_trace()
        for k, v in pretrained_dict.items():
            if (k in model_dict):
                print("load:%s"%k)
                #pdb.set_trace()
            else:
                print("jump over:%s"%k)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()
        
    def forward(self, x):
        if x.shape[2] == 1:
            x_in = torch.cat([x, x, x], 2)
        else:
            x_in = x

        origin_shape = x_in.shape
        x_in = x_in.view(-1, * origin_shape[2:])

        if x.shape[2] == 4:
            x_in = self.first_conv(x_in)

        fmap=self.backbone.feat_net(x_in)
        corr_vol = self.nlnet(fmap, origin_shape)
        out = self.backbone.decoder(corr_vol)
        return out
