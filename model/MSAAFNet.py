from torch import nn
from torch import Tensor
import torch
import torchvision.models as models
import torch.nn.functional as F

__all__ = ["MSAAFNet"]

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}

mob_conv1_2 = mob_conv2_2 = mob_conv3_3 = mob_conv4_3 = mob_conv5_3 = None

def conv_1_2_hook(module, input, output):
    global mob_conv1_2
    mob_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global mob_conv2_2
    mob_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global mob_conv3_3
    mob_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global mob_conv4_3
    mob_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global mob_conv5_3
    mob_conv5_3 = output
    return None

def US2(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear')
def US4(x):
    return F.interpolate(x, scale_factor=4, mode='bilinear')
def US8(x):
    return F.interpolate(x, scale_factor=8, mode='bilinear')
def US16(x):
    return F.interpolate(x, scale_factor=16, mode='bilinear')

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.mbv = models.mobilenet_v2(pretrained=True).features

        self.mbv[1].register_forward_hook(conv_1_2_hook)
        self.mbv[3].register_forward_hook(conv_2_2_hook)
        self.mbv[6].register_forward_hook(conv_3_3_hook)
        self.mbv[13].register_forward_hook(conv_4_3_hook)
        self.mbv[17].register_forward_hook(conv_5_3_hook)

    def forward(self, x: Tensor) -> Tensor:
        global mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3
        self.mbv(x)

        return mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3

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
        x = self.relu(x)
        return x

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):

        return self.reduce(x)


class Edge_Enhancement(nn.Module):
    def __init__(self, channel):
        super(Edge_Enhancement, self).__init__()
        self.Conv1 = BasicConv2d(channel + 1, channel, 3, padding=1)
        self.Conv2 = BasicConv2d(2, 1, 3, padding=1)
        self.Conv3 = BasicConv2d(channel + 1, channel, 3, padding=1)
        self.Conv4 = BasicConv2d(3, 1, 3, padding=1)
        self.Conv5 = BasicConv2d(channel + 1, channel, 3, padding=1)
        self.Conv6 = BasicConv2d(4, 1, 3, padding=1)
        self.Conv7 = BasicConv2d(channel + 1, channel, 3, padding=1)
        self.Conv8 = BasicConv2d(5, 1, 3, padding=1)
        self.Conv9 = BasicConv2d(channel + 1, channel, 3, padding=1)
        self.edg_pred = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.sa = SpatialAttention()

    def forward(self, x, x_e1=None, x_e2=None, x_e3=None, x_e4=None):
        if x_e1 is None and x_e2 is None and x_e3 is None and x_e4 is None:
            x_sa = x * self.sa(x)
            e_pred = self.edg_pred(x_sa)
            e_fuse = e_pred
            x = self.Conv1(torch.cat((e_pred, x), 1)) + x
        else:
            if x_e1 is not None and x_e2 is None and x_e3 is None and x_e4 is None:
                x_sa = x * self.sa(x)
                e_pred = self.edg_pred(x_sa)
                x_e1 = F.interpolate(x_e1, size= x.size()[2:], mode='bilinear', align_corners=True)
                e_fuse = self.Conv2(torch.cat((e_pred, x_e1), 1))
                x = self.Conv3(torch.cat((e_fuse, x), 1)) + x # Feature + Masked Edge information
            else:
                if x_e1 is not None and x_e2 is not None and x_e3 is None and x_e4 is None:
                    x_sa = x * self.sa(x)
                    e_pred = self.edg_pred(x_sa)
                    x_e1 = F.interpolate(x_e1, size=x.size()[2:], mode='bilinear', align_corners=True)
                    x_e2 = F.interpolate(x_e2, size=x.size()[2:], mode='bilinear', align_corners=True)
                    e_fuse = self.Conv4(torch.cat((e_pred, x_e1, x_e2), 1))
                    x = self.Conv5(torch.cat((e_fuse, x), 1)) + x  # Feature + Masked Edge information
                else:
                    if x_e1 is not None and x_e2 is not None and x_e3 is not None and x_e4 is None:
                        x_sa = x * self.sa(x)
                        e_pred = self.edg_pred(x_sa)
                        x_e1 = F.interpolate(x_e1, size=x.size()[2:], mode='bilinear', align_corners=True)
                        x_e2 = F.interpolate(x_e2, size=x.size()[2:], mode='bilinear', align_corners=True)
                        x_e3 = F.interpolate(x_e3, size=x.size()[2:], mode='bilinear', align_corners=True)
                        e_fuse = self.Conv6(torch.cat((e_pred, x_e1, x_e2, x_e3), 1))
                        x = self.Conv7(torch.cat((e_fuse, x), 1)) + x  # Feature + Masked Edge information
                    else:
                        if x_e1 is not None and x_e2 is not None and x_e3 is not None and x_e4 is not None:
                            x_sa = x * self.sa(x)
                            e_pred = self.edg_pred(x_sa)
                            x_e1 = F.interpolate(x_e1, size=x.size()[2:], mode='bilinear', align_corners=True)
                            x_e2 = F.interpolate(x_e2, size=x.size()[2:], mode='bilinear', align_corners=True)
                            x_e3 = F.interpolate(x_e3, size=x.size()[2:], mode='bilinear', align_corners=True)
                            x_e4 = F.interpolate(x_e4, size=x.size()[2:], mode='bilinear', align_corners=True)
                            e_fuse = self.Conv8(torch.cat((e_pred, x_e1, x_e2, x_e3, x_e4), 1))
                            x = self.Conv9(torch.cat((e_fuse, x), 1)) + x

        return x, e_fuse, e_pred

class MSAAF(nn.Module):
    def __init__(self, channel):
        super(MSAAF, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(channel, channel, kernel_size=(9, 1), padding=(4, 0))
        )

        self.conv_cat1 = BasicConv2d(4 * channel, channel, 3, padding=1)
        self.conv_cat2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_cat3 = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.conv_cat4 = BasicConv2d(4 * channel, channel, 3, padding=1)
        self.conv_cat5 = BasicConv2d(5 * channel, channel, 3, padding=1)
        self.conv3_3 = BasicConv2d(channel, channel, 3, padding=1)

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
        self.fc = nn.Sequential(nn.Conv2d(channel, channel // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(channel // 2, channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, e_fuse, a_gca5=None, a_gca4=None, a_gca3=None, a_gca2=None):
        if a_gca5 is None and a_gca4 is None and a_gca3 is None and a_gca2 is None:
            x0 = self.conv3_3(x)
            a_cse = self.ca(x0) * (self.sa(x0) + self.sigmoid(e_fuse))
            a_cse = self.fc(a_cse)
            a_gca = F.softmax(a_cse, dim=1)
            x3_3 = self.branch1(x0) * a_gca
            x5_5 = self.branch2(x0) * a_gca
            x7_7 = self.branch3(x0) * a_gca
            x9_9 = self.branch4(x0) * a_gca
            s_msaaf = self.conv_cat1(torch.cat((x3_3, x5_5, x7_7, x9_9), 1)) + x
        else:
            if a_gca5 is not None and a_gca4 is None and a_gca3 is None and a_gca2 is None:
                x0 = self.conv3_3(x)
                a_cse = self.ca(x0) * (self.sa(x0) + self.sigmoid(e_fuse))
                a_cse = self.fc(a_cse)
                a = F.softmax(a_cse, dim=1)
                a_gca = self.conv_cat2(torch.cat((a, US2(a_gca5)), 1))
                x3_3 = self.branch1(x0) * a_gca
                x5_5 = self.branch2(x0) * a_gca
                x7_7 = self.branch3(x0) * a_gca
                x9_9 = self.branch4(x0) * a_gca
                s_msaaf = self.conv_cat1(torch.cat((x3_3, x5_5, x7_7, x9_9), 1)) + x
            else:
                if a_gca5 is not None and a_gca4 is not None and a_gca3 is None and a_gca2 is None:
                    x0 = self.conv3_3(x)
                    a_cse = self.ca(x0) * (self.sa(x0) + self.sigmoid(e_fuse))
                    a_cse = self.fc(a_cse)
                    a = F.softmax(a_cse, dim=1)
                    a_gca = self.conv_cat3(torch.cat((a, US4(a_gca5), US2(a_gca4)), 1))
                    x3_3 = self.branch1(x0) * a_gca
                    x5_5 = self.branch2(x0) * a_gca
                    x7_7 = self.branch3(x0) * a_gca
                    x9_9 = self.branch4(x0) * a_gca
                    s_msaaf = self.conv_cat1(torch.cat((x3_3, x5_5, x7_7, x9_9), 1)) + x
                else:
                    if a_gca5 is not None and a_gca4 is not None and a_gca3 is not None and a_gca2 is None:
                        x0 = self.conv3_3(x)
                        a_cse = self.ca(x0) * (self.sa(x0) + self.sigmoid(e_fuse))
                        a_cse = self.fc(a_cse)
                        a = F.softmax(a_cse, dim=1)
                        a_gca = self.conv_cat4(torch.cat((a, US8(a_gca5), US4(a_gca4), US2(a_gca3)), 1))
                        x3_3 = self.branch1(x0) * a_gca
                        x5_5 = self.branch2(x0) * a_gca
                        x7_7 = self.branch3(x0) * a_gca
                        x9_9 = self.branch4(x0) * a_gca
                        s_msaaf = self.conv_cat1(torch.cat((x3_3, x5_5, x7_7, x9_9), 1)) + x
                    else:
                        if a_gca5 is not None and a_gca4 is not None and a_gca3 is not None and a_gca2 is not None:
                            x0 = self.conv3_3(x)
                            a_cse = self.ca(x0) * (self.sa(x0) + self.sigmoid(e_fuse))
                            a_cse = self.fc(a_cse)
                            a = F.softmax(a_cse, dim=1)
                            a_gca = self.conv_cat5(torch.cat((a, US16(a_gca5), US8(a_gca4), US4(a_gca3), US2(a_gca2)), 1))
                            x3_3 = self.branch1(x0) * a_gca
                            x5_5 = self.branch2(x0) * a_gca
                            x7_7 = self.branch3(x0) * a_gca
                            x9_9 = self.branch4(x0) * a_gca
                            s_msaaf = self.conv_cat1(torch.cat((x3_3, x5_5, x7_7, x9_9), 1)) + x
        return s_msaaf, a_gca

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MSAAFNet(nn.Module):
    def __init__(self, channel=32):

        super(MSAAFNet, self).__init__()
        self.encoder = MobileNet()
        self.reduce_sal1 = Reduction(16, channel)
        self.reduce_sal2 = Reduction(24, channel)
        self.reduce_sal3 = Reduction(32, channel)
        self.reduce_sal4 = Reduction(96, channel)
        self.reduce_sal5 = Reduction(320, channel)

        self.msaaf5 = MSAAF(channel)
        self.msaaf4 = MSAAF(channel)
        self.msaaf3 = MSAAF(channel)
        self.msaaf2 = MSAAF(channel)
        self.msaaf1 = MSAAF(channel)

        self.ee1 = Edge_Enhancement(channel)
        self.ee2 = Edge_Enhancement(channel)
        self.ee3 = Edge_Enhancement(channel)
        self.ee4 = Edge_Enhancement(channel)
        self.ee5 = Edge_Enhancement(channel)

        self.S1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.S_conv1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv2 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv3 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.S_conv4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.trans_conv1 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        self.trans_conv2 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv3 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv4 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        size = x.size()[2:]
        x_sal1, x_sal2, x_sal3, x_sal4, x_sal5 = self.encoder(x)

        x_sal1 = self.reduce_sal1(x_sal1)
        x_sal2 = self.reduce_sal2(x_sal2)
        x_sal3 = self.reduce_sal3(x_sal3)
        x_sal4 = self.reduce_sal4(x_sal4)
        x_sal5 = self.reduce_sal5(x_sal5)

        s_ee1, edge_fuse1, edg1 = self.ee1(x_sal1)
        s_ee2, edge_fuse2, edg2 = self.ee2(x_sal2, edg1)
        s_ee3, edge_fuse3, edg3 = self.ee3(x_sal3, edg1, edg2)
        s_ee4, edge_fuse4, edg4 = self.ee4(x_sal4, edg1, edg2, edg3)
        s_ee5, edge_fuse5, edg5 = self.ee5(x_sal5, edg1, edg2, edg3, edg4)

        s_msaaf5, a_gca5 = self.msaaf5(s_ee5, edge_fuse5)
        s_msaaf4, a_gca4 = self.msaaf4(s_ee4, edge_fuse4, a_gca5)
        s_msaaf3, a_gca3 = self.msaaf3(s_ee3, edge_fuse3, a_gca5, a_gca4)
        s_msaaf2, a_gca2 = self.msaaf2(s_ee2, edge_fuse2, a_gca5, a_gca4, a_gca3)
        s_msaaf1, a_gca1 = self.msaaf1(s_ee1, edge_fuse1, a_gca5, a_gca4, a_gca3, a_gca2)

        sal4 = self.S_conv1(torch.cat([s_msaaf4, self.trans_conv1(s_msaaf5)], dim=1))
        sal3 = self.S_conv2(torch.cat([s_msaaf3, self.trans_conv2(sal4)], dim=1))
        sal2 = self.S_conv3(torch.cat([s_msaaf2, self.trans_conv3(sal3)], dim=1))
        sal1 = self.S_conv4(torch.cat([s_msaaf1, self.trans_conv4(sal2)], dim=1))

        sal_out = self.S1(sal1)
        sal2 = self.S2(sal2)
        sal3 = self.S3(sal3)
        sal4 = self.S4(sal4)
        sal5 = self.S5(s_msaaf5)

        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        sal2 = F.interpolate(sal2, size=size, mode='bilinear', align_corners=True)
        sal3 = F.interpolate(sal3, size=size, mode='bilinear', align_corners=True)
        sal4 = F.interpolate(sal4, size=size, mode='bilinear', align_corners=True)
        sal5 = F.interpolate(sal5, size=size, mode='bilinear', align_corners=True)
        edg1 = F.interpolate(edg1, size=size, mode='bilinear', align_corners=True)
        edg2 = F.interpolate(edg2, size=size, mode='bilinear', align_corners=True)
        edg3 = F.interpolate(edg3, size=size, mode='bilinear', align_corners=True)
        edg4 = F.interpolate(edg4, size=size, mode='bilinear', align_corners=True)
        edg5 = F.interpolate(edg5, size=size, mode='bilinear', align_corners=True)

        return sal_out, self.sigmoid(sal_out), edg1, sal2, self.sigmoid(sal2), edg2, sal3, self.sigmoid(
            sal3), edg3, sal4, self.sigmoid(sal4), edg4, sal5, self.sigmoid(sal5), edg5