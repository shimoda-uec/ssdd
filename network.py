import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import cv2
import numpy as np
import time
from torch.autograd import Variable

class Conv2dbnPR(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(Conv2dbnPR,self).__init__()

        self.rpad = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,0,dilation, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
    def forward(self,x):
        x = self.rpad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x
        
class Conv2dbn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(Conv2dbn,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x
        
class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01)
        self.rpad = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bnd = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.rpad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bnd(identity)
        out += identity
        out = self.relu(out)
        return out

class PredictDiffHead(nn.Module):
    def __init__(self, config,cln=21, in_channel=256, dr_rate_a=0.5, in_channel2=128):
        super(PredictDiffHead, self).__init__()
        self.config=config
        chn=256
        self.conv1ab = Conv2dbnPR(in_channel2 + in_channel, chn, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        xa_in, xb_in = inputs
        xab=torch.cat((xa_in,xb_in),dim=1)
        xab=self.conv1ab(xab)
        return xab


class PredictDiff(nn.Module):
    def __init__(self, config, cln=21, in_channel=256, in_channel2=128, dr_rate_d=0.5):
        super(PredictDiff, self).__init__()
        self.config=config
        chn=256
        self.conv1c = Conv2dbnPR(cln, chn, kernel_size=1, stride=1, padding=0)
        self.conv1abc = Bottleneck(chn*2,chn,kernel_size=3,padding=1)
        self.pred_abc = nn.Conv2d(chn,1,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self, inputs):
        xab, xc_in =inputs
        xc=self.conv1c(xc_in)
        xabc=torch.cat((xab,xc),dim=1)
        xabc=self.conv1abc(xabc)
        xabc=self.pred_abc(xabc)
        return xabc


class SegmentationPsa(nn.Module):
    def __init__(self, config, num_classes, in_channel=4096, middle_channel=512, scale=8):
        super(SegmentationPsa, self).__init__()
        self.config=config
        self.seg1 = Conv2dbnPR(in_channel,middle_channel,3,1,padding=12, dilation=12, bias=True)
        self.rpad = nn.ReflectionPad2d(12)
        self.seg2 = nn.Conv2d(middle_channel,21,kernel_size=3,stride=1,padding=0,dilation=12, bias=True)
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear')
    def forward(self, inputs):
        x=inputs
        seg_head=self.seg1(x)
        x=self.rpad(seg_head)
        x=self.seg2(x)
        seg_head=self.upsample(seg_head)
        x=self.upsample(x)
        return x, seg_head
