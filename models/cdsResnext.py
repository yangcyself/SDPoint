# -^- coding:utf-8 -^-
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

__all__ = ['cdsResNeXt', 'cdsresnext50', 'cdsresnext101', 'cdsresnext152']

ModelDebug = False
blockID = 0
class pipeConv(nn.Module):
    # this module maintains a mask of whether input channel is downsampled 
    #   and whether output channel should be downsampled
    masks = {} # the dict preserving all the masks  layerId -> maks
    def __init__(self, in_maskid, out_maskid, *args, **kwargs):
        super(pipeConv,self).__init__(*args, **kwargs)
        self.in_maskid = in_maskid
        self.out_maskid = out_maskid
        self.conv = nn.Conv2d(*args, **kwargs)
        self.upspl =  nn.Upsample(scale_factor=2, mode='nearest')
        self.dnspl = nn.AvgPool2d(2)

    def forward(self,x):
        if(ModelDebug):
            print("xshape:",x.shape)
        if(x.shape[2]%2==1):
            x = F.pad(input=x, pad=(0, 1 , 0, 1), mode='constant', value=0)
        halfshape = (int(x.shape[2]/2), int(x.shape[3]/2))
        in_mask = pipeConv.masks[self.in_maskid]
        out_mask = pipeConv.masks[self.out_maskid]
        x_s = (x * (1-in_mask).view(1,-1,1,1))[:,:,:halfshape[0],:halfshape[1]]
        x = x * in_mask.view(1,-1,1,1) # the feature maps which have larger size
        x = self.conv(x)
        x_s = self.conv(x_s)
        
        o_l = x + self.upspl(x_s)
        o_s = self.dnspl(x) + x_s

        o_s = F.pad(input=o_s, pad=(0, o_l.shape[2] - o_s.shape[2], 0, o_l.shape[3] - o_s.shape[3]), mode='constant', value=0)
        del(x)
        del(x_s)
        torch.cuda.empty_cache()
        out = o_l*out_mask.view(1,-1,1,1) + o_s*(1-out_mask).view(1,-1,1,1)
        return out




class Bottleneck(nn.Module):
	expansion = 4
			# block(self.inplanes, planes, self.base_width, self.cardinality, stride, downsample)
	def __init__(self, inplanes, planes, base_width, cardinality, stride=1, pipeMaskIds = None ):
		super(Bottleneck, self).__init__()
		D = int(math.floor(planes * (base_width / 64)))
		C = cardinality
		self.shape = (inplanes, D*C,planes)
		assert (pipeMaskIds is not None) 
		self.maskid = pipeMaskIds[-1] # the input mask of the first conv in bottleneck
		
		pipeConv.masks[pipeMaskIds[-1]] = torch.Tensor(np.array([1]*inplanes)).cuda()
		self.conv1 = pipeConv(self.maskid,self.maskid+1, inplanes, D*C, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(D*C)

		pipeMaskIds.append(self.maskid+1)
		pipeConv.masks[pipeMaskIds[-1]] = torch.Tensor(np.array([1]*D*C)).cuda()
		self.conv2 = pipeConv(self.maskid+1, self.maskid+2 ,D*C, D*C, kernel_size=3, stride=stride,
							   padding=1, groups=C, bias=False)
		self.bn2 = nn.BatchNorm2d(D*C)

		pipeMaskIds.append(self.maskid+2)
		pipeConv.masks[pipeMaskIds[-1]] = torch.Tensor(np.array([1]*D*C)).cuda()
		self.conv3 = pipeConv(self.maskid+2, self.maskid+3 ,D*C, planes * self.expansion, kernel_size=1, bias=False)

		pipeMaskIds.append(self.maskid+3)

		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.stride = stride
		self.inplanes = inplanes
		global blockID
		self.blockID = blockID
		blockID += 1

	def forward(self, x):
    		
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		out += residual
		out = self.relu(out)

		return out


class CdsResNeXt(nn.Module):

	def __init__(self, block, layers, base_width=4, cardinality=32, num_classes=1000):
		self.cardinality = cardinality
		self.base_width = base_width
		self.inplanes = 64
		super(CdsResNeXt, self).__init__()
		global blockID
		blockID = 1

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, #没明白这个conv1是干啥用的
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.allblocks=[] # 
		self.pipeMaskIds = [0]
		self.layer1 = self._make_layer(block, 64, layers[0],pipeMaskIDs = self.pipeMaskIds)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2,pipeMaskIDs = self.pipeMaskIds)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2,pipeMaskIDs = self.pipeMaskIds)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2,pipeMaskIDs = self.pipeMaskIds)
		pipeConv.masks[self.pipeMaskIds[-1]] = torch.Tensor(np.array([1]*512*4)).cuda() # planes * self.expansion
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		self.blockID = blockID
		self.size_after_maxpool = None

		for m in self.modules():
			if isinstance(m, pipeConv):
				m = m.conv
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1,pipeMaskIDs = None):
		layers = []
		assert (pipeMaskIDs is not None)
		layers.append(block(self.inplanes, planes, self.base_width, self.cardinality, stride, pipeMaskIDs))
		self.allblocks.append(layers[-1]) # add blocks according to the blockID

		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, self.base_width, self.cardinality, pipeMaskIDs))
			self.allblocks.append(layers[-1])

		return nn.Sequential(*layers)

		

	def forward(self, x, blockID=None, ratio=None):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
	

	# def step(self, x, blockID=0, ratio = 1 ):
	# 	if(blockID==0): # the first step 
	# 		x = self.conv1(x)
	# 		x = self.bn1(x)
	# 		x = self.relu(x)
	# 		return x
	# 	if(blockID==self.blockID):
	# 		x = self.avgpool(x)
	# 		x = x.view(x.size(0), -1)
	# 		x = self.fc(x)
	# 		return x
	# 	self.allblocks[blockID].downsampling_ratio = ratio
	# 	return self.allblocks[blockID].forward(x)

    		

def cdsresnext50(pretrained=False, **kwargs):
	"""Constructs a CdsResNeXt-50 model.
	"""
	model = CdsResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
	return model


def cdsresnext101(pretrained=False, **kwargs):
	"""Constructs a CdsResNeXt-101 model.
	"""
	model = CdsResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
	return model


def cdsresnext152(pretrained=False, **kwargs):
	"""Constructs a CdsResNeXt-152 model.
	"""
	model = CdsResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
	return model
