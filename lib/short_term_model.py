import torch
import torch.nn as nn

from lib.short_term_pyramid import CorrealationBlock
from lib.pvtv2_afterTEM import Network

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
        self.nl_layer1 = CorrealationBlock(in_planes*2, None, False, scale=2)
        self.nl_layer2 = CorrealationBlock(in_planes*2, None, False, scale=4)
        self.nl_layer3 = CorrealationBlock(in_planes*2, None, False, scale=8)

        self.conv1 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)
        self.conv2 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)
        self.conv3 = BasicConv2d(in_planes*2, in_planes, 3, padding=1)

    def forward(self, fea1, fea2):
        #pdb.set_trace()
        out1 = self.conv1(self.nl_layer1(torch.cat([fea1[0], fea2[0]], dim=1)))
        out2 = self.conv2(self.nl_layer2(torch.cat([fea1[1], fea2[1]], dim=1)))
        out3 = self.conv3(self.nl_layer3(torch.cat([fea1[2], fea2[2]], dim=1)))
        return out1, out2, out3

class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.args = args
        self.backbone = Network(pvtv2_pretrained=self.args.pvtv2_pretrained, imgsize=self.args.trainsize)
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
        if self.args.pretrained_cod10k is not None:
            self.load_backbone(self.args.pretrained_cod10k )

        self.nlnet = NonLocalNet(in_planes=32, pyramid_type='conv')

        self.fusion_conv = nn.Sequential(nn.Conv2d(2, 32, 3, 1, 1),
                                         nn.Conv2d(32, 32, 3, 1, 1),
                                         nn.Conv2d(32, 1, 3, 1, 1),
            )

        self.freeze_bn()

    def load_backbone(self, pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = self.state_dict()
        print("Load pretrained cod10k parameters from {}".format(pretrained))
        for k, v in pretrained_dict.items():
            if (k in model_dict):
                print("load:%s"%k)
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
        image1, image2, image3 = x[0],x[1],x[2]
        fmap1=self.backbone.feat_net(image1)
        fmap2=self.backbone.feat_net(image2)
        fmap3=self.backbone.feat_net(image3)

        corr_vol12 = self.nlnet(fmap1, fmap2)
        corr_vol13 = self.nlnet(fmap1, fmap3)

        out12 = self.backbone.decoder(corr_vol12)
        out13 = self.backbone.decoder(corr_vol13)

        concated = torch.cat([out12[-1], out13[-1]], dim=1)
        out = self.fusion_conv(concated)

        return out12, out13, out
