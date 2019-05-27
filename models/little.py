# -^- coding:utf-8 -^-
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import random
import torch.utils.model_zoo as model_zoo
import numpy as np

class pipeConv(nn.Module):
    # this module maintains a mask of whether input channel is downsampled 
    #   and whether output channel should be downsampled
    def __init__(self, inplanes, outplanes,*args, **kwargs):
        super(pipeConv,self).__init__(*args, **kwargs)
        self.in_mask = torch.Tensor(np.array([1]*inplanes)).cuda()
        self.out_mask = torch.Tensor(np.array([1]*outplanes)).cuda()
        self.conv = nn.Conv2d(inplanes, outplanes , kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upspl =  nn.Upsample(scale_factor=2, mode='nearest')
        self.dnspl = nn.AvgPool2d(2)
    def forward(self,x):
        print("xshape:",x.shape)
        halfshape = (int(x.shape[2]/2), int(x.shape[3]/2))
        print(torch.cuda.memory_allocated())
        # x_l = x * self.in_mask.view(1,-1,1,1) # the feature maps which have larger size
        # x_s = (x * (1-self.in_mask).view(1,-1,1,1))[:,:,:halfshape[0],:halfshape[1]]
        x_s = (x * (1-self.in_mask).view(1,-1,1,1))[:,:,:halfshape[0],:halfshape[1]]
        print(torch.cuda.memory_allocated())
        x = x * self.in_mask.view(1,-1,1,1) # the feature maps which have larger size
        print(torch.cuda.memory_allocated())

        print(torch.cuda.memory_allocated())
        x = self.conv(x)
        
        print(torch.cuda.memory_allocated())
        x_s = self.conv(x_s)
        # print("x_lshape:",x_l.shape)
        # print("x_sshape:",x_s.shape)
        # print("upsplxshape:",self.upspl(x_s).shape)
        # print("dnsplxshape:",self.dnspl(x_l).shape)
        
        print(torch.cuda.memory_allocated())
        o_l = x + self.upspl(x_s)
        o_s = self.dnspl(x) + x_s
        # o_l = x_l + self.upspl(x_s)
        # o_s = self.dnspl(x_l) + x_s

        print(torch.cuda.memory_allocated())
        o_s = F.pad(input=o_s, pad=(0, o_l.shape[2] - o_s.shape[2], 0, o_l.shape[3] - o_s.shape[3]), mode='constant', value=0)
        # print("osshape",o_s.shape)
        del(x)
        del(x_s)
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated())
        out = o_l*self.out_mask.view(1,-1,1,1) + o_s*(1-self.out_mask).view(1,-1,1,1)
        out = self.bn(out)
        out = self.relu(out)
        # print("outshape",out.shape)
        return out

class pipeNet(nn.Module):
    def __init__(self,out_num, *args, **kwargs):
        super(pipeNet,self).__init__(*args, **kwargs)
        self.cnn1 = pipeConv(3,96)
        self.cnn2 = pipeConv(96,256)
        # self.cnn3 = pipeConv(256,384)
        # self.cnn4 = pipeConv(384,384)
        # self.cnn5 = pipeConv(384,1024)
        self.cnn5 = pipeConv(256,1024)
        self.fc1 = nn.Linear(1024,out_num)
        self.pool = nn.AvgPool2d(2)
        self.final_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.pool(x)
        # x = self.cnn3(x)
        # x = self.cnn4(x)
        # x = self.pool(x)
        x = self.cnn5(x)
        x = self.final_pool(x)

        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x



    
