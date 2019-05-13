# -^- coding : utf-8 -^-
"""
This function is used to:
 1. hook out features maps from a model
 2. calculate the entropy 
 3. calculate the activation
"""


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# from module import Net
import pickle as pkl


class activationHooker:
    def __init__(self,model,classNum = 100):
        # self.hookdist = {}
        self.activations = {}
        self.groundTruth = []
        for i in range(classNum):
            self.activations[i] = {}
        self.model = model
        self.model.apply(lambda m: self.reg_hook(m,lambda m,inp,outp: self.hook(m,inp,outp,self.activations)))

    def reg_hook(self,m,hook):
        m.register_forward_hook(hook)

    def hook(self,m,inp,outp,stor):
        assert(len(self.groundTruth)==outp.shape[0])
        if(isinstance(m, torch.nn.modules.conv.Conv2d)):
            for n, atv in zip(self.groundTruth,outp):
                if(m in stor[n].keys()):
                    act = stor[n][m]
                    act += atv.cpu().numpy()
                else:
                    act = atv.cpu().numpy()
                stor[n][m] = act

    def dataloaders(self,data):
        #return dataloaders of different number (number,dataloader)
        # datas = {}
        # for dt, tg in data:
        #     datas{tg}
        pass
    
    def analysisACT(self,loader,device):
        self.model.eval()
        mask2 = mask1 = torch.ones([100]).to(device)
        count = 0
        for i, (batch, label) in enumerate(loader):
            batch = batch.to(device)
            label = label.to(device)
                # print(target)
            # self.hookdist = self.activations[int(target.numpy())] # change the pointer to make ti
            self.groundTruth = label
                
            output = self.model(data, mask1, mask2)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            count+=1
                # print("a")
        tempdict = {}
        # for lay in self.hookdist.keys():
        for i in range(10):
            tempdict[i] = {}
            for name, act in self.model.named_children():
                tempdict[i][name] = self.activations[i][act]/count
                # tempdict[lay] = self.hookdist[lay]/count
        self.hookdist = {}
        print(tempdict)
        
        return tempdict

if __name__ == "__main__":
    test_path = "./mnist_data"
    use_cuda = torch.cuda.is_available()
    pin_memory = True
    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    dataset=DataSet(torch_v=0.4)
    test_loader = dataset.test_loader(test_path,pin_memory=pin_memory)

    model = torch.load("vgg16_model_mnist").to(device)
    hooker = activationHooker(model)
    tempdist = hooker.analysisACT(test_loader,device)
    with open("tempdist.pkl","wb") as f:
        pkl.dump(tempdist,f)

    # only these things are pickable
    # functions defined at the top level of a module (using def, not >lambda)
    # built-in functions defined at the top level of a module
    # classes that are defined at the top level of a module


