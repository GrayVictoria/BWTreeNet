import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Softmax
import numpy as np
from thop import profile
from thop import clever_format


class ResDoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, mid_channels=None):
        super(ResDoubleConv, self).__init__()

        if not mid_channels:
            mid_channels = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x = x + residual
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResDoubleConv(in_channels, out_channels,
                          mid_channels=out_channels//2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down_Att(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2)
        self.maxpool = nn.MaxPool2d(2)

        self.singleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )
        # self.singleconv = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        self.doubleconv = ResDoubleConv(
            out_channels, out_channels, mid_channels=out_channels)

    def forward(self, x, x_ori):
        x_sharp = self.singleconv(x)
        x_att = (x_ori/255.0)
        x = x_sharp*x_att
        x = torch.cat([x, x_sharp], dim=1)
        x_skip = self.doubleconv(x)
        # x = self.maxpool(x_skip)
        x = self.avgpool(x_skip)
        return x, x_skip


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResDoubleConv(
                in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResDoubleConv(
                in_channels, out_channels, in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up_Out(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels+out_channels, out_channels,
                          3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax()
        

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=True, expansion=4):
        super(SEBottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places *
                      self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        self.se = SELayer(places * self.expansion, 8)
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # print('se', x.shape)
        out = self.bottleneck(x)
        out = self.se(out)
        # print('out', out.shape)
        if self.downsampling:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class SharpConnect(nn.Module):
    # k1 = np.array([[0.0625, 0.125, 0.0625],
    #                [0.125,  0.25, 0.125],
    #                [0.0625, 0.125, 0.0625]])

    # # Sharpening Spatial Kernel, used in paper
    # k2 = np.array([[-1, -1, -1],
    #                [-1,  8, -1],
    #                [-1, -1, -1]])

    # k3 = np.array([[0, -1, 0],
    #                [-1,  5, -1],
    #                [0, -1, 0]])
    def __init__(self, in_ch_ori, in_ch, out_ch):
        super(SharpConnect, self).__init__()
        if in_ch == 128:
            Co, Ho, Wo = [1, 250, 250]
            Cf, Hf, Wf = [64, 250, 250]
        if in_ch == 64:
            Co, Ho, Wo = [1, 500, 500]
            Cf, Hf, Wf = [32, 500, 500]
        elif in_ch == 32:
            Co, Ho, Wo = [1, 1000, 1000]
            Cf, Hf, Wf = [32, 1000, 1000]

        KenForOri = [[[[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]]]
        KenForOri = torch.FloatTensor(KenForOri).expand(
            in_ch_ori, in_ch_ori, 3, 3)

        KenForFtr = [[[[0, -1, 0],
                       [-1,  4, -1],
                       [0, -1, 0]]]]
        KenForFtr = torch.FloatTensor(KenForFtr).expand(out_ch, in_ch, 3, 3)

        self.SharpOri = nn.Conv2d(
            in_ch_ori, in_ch_ori, (3, 3), padding=1, bias=False)
        self.SharpOri.weight.data = KenForOri

        self.SharpFeature = nn.Conv2d(
            in_ch, out_ch, (3, 3), padding=1, bias=False)
        self.SharpFeature.weight.data = KenForFtr

        self.ConvOri = nn.Sequential(
            nn.Conv2d(in_ch_ori, out_ch//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(),
            nn.Conv2d(out_ch//2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.lyo = nn.LayerNorm([Co, Ho, Wo])
        self.lyf = nn.LayerNorm([Cf, Hf, Wf])
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x_ori, x_ftr):
        N, C, H, W = x_ori.shape
        N, C, H, W = x_ftr.shape
        x_ori = self.SharpOri(x_ori)
        x_ori = self.lyo(x_ori)
        x_ori = self.ConvOri(x_ori)

        x_ftr = self.SharpFeature(x_ftr)
        x_ftr = self.lyf(x_ftr)
        x_ftr = self.relu(x_ftr)

        x = torch.cat([x_ftr, x_ori], dim=1)
        return x


class BWTreeNet(nn.Module):
    def __init__(self, n_class, bilinear=False):
        super(BWTreeNet, self).__init__()
        self.n_channels = 1
        self.n_class = n_class
        self.bilinear = bilinear

        self.down1 = Down_Att(self.n_channels, 64)
        # self.down2 = Down_Att(64, 128)
        self.down2 = Down_Att(64, 128)
        # self.down3 = Down_Att(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512 // factor)
        self.down5 = Down(512, 1024//factor)

        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up_Out(64, 32, bilinear)

        self.avgpool2 = nn.AvgPool2d(2)
        self.avgpool3 = nn.AvgPool2d(2)

        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(2)

        self.sharp3 = SharpConnect(1, 128, 64)
        self.sharp4 = SharpConnect(1, 64, 32)
        self.sharp5 = SharpConnect(1, 64, 16)

        self.se1 = SEBottleneck(512, 512, downsampling=True, expansion=1)
        self.se2 = SEBottleneck(256, 256, downsampling=True, expansion=1)
        self.se3 = SEBottleneck(128, 128, downsampling=True, expansion=1)
        self.se4 = SEBottleneck(64, 64, downsampling=True, expansion=1)

        self.outc = OutConv(32, n_class)

    def forward(self, x):
        # x_ori = torch.cat([x,x],dim = 1)
        # x_ori = torch.cat([x_ori,x],dim = 1)
        # x_ori = x
        x = 255-x
        x_11 = x
        x_a2 = self.avgpool2(x)
        x_a4 = self.avgpool3(x_a2)

        x_m2 = self.maxpool2(x)
        # x_m4 = self.maxpool3(x_m2)

        x1, x_s1 = self.down1(x, x_11)

        x2,x_s2 = self.down2(x1,x_m2)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)

        x = self.se1(x)

        x = self.up2(x, x3)
        x = self.se2(x)

        # print('x2', x2.shape)
        x2 = self.sharp3(x_a4, x2)

        x = self.up3(x, x2)
        x = self.se3(x)

        x1 = self.sharp4(x_a2, x1)

        x = self.up4(x, x1)
        x = self.se4(x)

        # x_o = self.sharp5(x_11, x_s1)

        # print('x1', x_s1.shape)
        # print('x1',x_o.shape)
        # print('x1', x.shape)
        x = self.up5(x, x_s1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = BWTreeNet(2).cuda()
    # model.apply(weights_init)
    for i in range(1):

        x = torch.randn((1, 1, 1000, 1000)).cuda()

        # for i in range(10):
        #     y0 = model(x)
        # print(y0)
        y0 = model(x)
        # print(y0.shape)
    flops, params = profile(model, inputs=(x,))
    macs, params = clever_format([flops, params], '%.4f')
    print(flops, params)
