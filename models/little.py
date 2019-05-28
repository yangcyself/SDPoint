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
        # print("xshape:",x.shape)
        if(x.shape[2]%2==1):
            x = F.pad(input=x, pad=(0, 1 , 0, 1), mode='constant', value=0)
        halfshape = (int(x.shape[2]/2), int(x.shape[3]/2))
        # print(torch.cuda.memory_allocated())
        # x_l = x * self.in_mask.view(1,-1,1,1) # the feature maps which have larger size
        # x_s = (x * (1-self.in_mask).view(1,-1,1,1))[:,:,:halfshape[0],:halfshape[1]]
        x_s = (x * (1-self.in_mask).view(1,-1,1,1))[:,:,:halfshape[0],:halfshape[1]]
        # print(torch.cuda.memory_allocated())
        x = x * self.in_mask.view(1,-1,1,1) # the feature maps which have larger size
        # print(torch.cuda.memory_allocated())

        # print(torch.cuda.memory_allocated())
        x = self.conv(x)
        
        # print(torch.cuda.memory_allocated())
        x_s = self.conv(x_s)
        # print("x_lshape:",x_l.shape)
        # print("x_sshape:",x_s.shape)
        # print("upsplxshape:",self.upspl(x_s).shape)
        # print("dnsplxshape:",self.dnspl(x_l).shape)
        
        # print(torch.cuda.memory_allocated())
        o_l = x + self.upspl(x_s)
        o_s = self.dnspl(x) + x_s
        # o_l = x_l + self.upspl(x_s)
        # o_s = self.dnspl(x_l) + x_s

        # print(torch.cuda.memory_allocated())
        o_s = F.pad(input=o_s, pad=(0, o_l.shape[2] - o_s.shape[2], 0, o_l.shape[3] - o_s.shape[3]), mode='constant', value=0)
        # print("osshape",o_s.shape)
        del(x)
        del(x_s)
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())
        out = o_l*self.out_mask.view(1,-1,1,1) + o_s*(1-self.out_mask).view(1,-1,1,1)
        out = self.bn(out)
        out = self.relu(out)
        # print("outshape",out.shape)
        return out

class pipeNet(nn.Module):
    def __init__(self,out_num,ds_probability = 1 ,*args, **kwargs):
        self.p = ds_probability
        super(pipeNet,self).__init__(*args, **kwargs)
        self.channel_num = [96,152,256,256,512]
        l_count = 0
        self.cnn1 = pipeConv(3,self.channel_num[l_count])
        l_count += 1
        # self.cnn1_1 = pipeConv(24,96)
        self.cnn2 = pipeConv(self.channel_num[l_count-1],self.channel_num[l_count])
        l_count += 1
        self.cnn3 = pipeConv(self.channel_num[l_count-1],self.channel_num[l_count])
        l_count += 1
        self.cnn4 = pipeConv(self.channel_num[l_count-1],self.channel_num[l_count])
        l_count += 1
        self.cnn5 = pipeConv(self.channel_num[l_count-1],self.channel_num[l_count])
        l_count += 1
        self.fc1 = nn.Linear(self.channel_num[l_count-1],out_num)
        l_count += 1
        self.pool = nn.MaxPool2d(2)
        self.avpool = nn.AvgPool2d(2)
        self.final_pool = nn.AdaptiveAvgPool2d(1)

        
    def forward(self,x):
        self.randomMask(self.p)
        x = self.avpool(x)
        x = self.cnn1(x)
        x = self.pool(x)
        # x = self.cnn1_1(x)
        # x = self.pool(x)
        x = self.cnn2(x)
        x = self.pool(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.pool(x)
        x = self.cnn5(x)
        x = self.final_pool(x)

        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x
    
    def randomMask(self,p):
        # np.random.binomial(1,p,(,))
        count = 0
        mask = np.random.binomial(1,p,(self.channel_num[count],))
        self.cnn1.out_mask = torch.Tensor(mask).cuda()
        self.cnn2.in_mask = torch.Tensor(mask).cuda()
        count += 1
        mask = np.random.binomial(1,p,(self.channel_num[count],))
        self.cnn2.out_mask = torch.Tensor(mask).cuda()
        self.cnn3.in_mask = torch.Tensor(mask).cuda()
        count += 1

        mask = np.random.binomial(1,p,(self.channel_num[count],))
        self.cnn3.out_mask =torch.Tensor(mask).cuda()
        self.cnn4.in_mask = torch.Tensor(mask).cuda()
        count += 1

        mask = np.random.binomial(1,p,(self.channel_num[count],))
        self.cnn4.out_mask = torch.Tensor(mask).cuda()
        self.cnn5.in_mask = torch.Tensor(mask).cuda()
        count += 1






    
