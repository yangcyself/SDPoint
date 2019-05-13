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
import sys
import os

# from module import Net
import pickle as pkl


class activationHooker:
    
    def __init__(self,model,targetLayers=[], outputdir = "output"):
        # self.hookdist = {}
        self.activations = {} # save one list of featuremaps for each layer
        self.model = model
        # targetLayers, the layer to which the hook is added
        print("Total Layers: ",model.allblocks[-1].blockID
                ,"hooked: ",targetLayers)
        for i in targetLayers:
            self.activations[i] = []
            self.reg_hook(self.model.allblocks[i],self.hook,self.activations[i])
        self.outputCount = 0
        self.outputdir = outputdir
    def reg_hook(self,m,hook,*args):
        m.register_forward_hook(lambda a,b,c : hook(a,b,c,*args))

    def hook(self,m,inp,outp,stor):
        self.checkBuffer()
        stor.append(outp.cpu().detach().numpy())

    def savebuffer(self):
        with open(os.path.join(self.outputdir,"%d.pkl"%self.outputCount),"wb") as f:
            pkl.dump(self.activations,f)

    def checkBuffer(self):
        # save the stored values into file
        if(sys.getsizeof(self.activations) > 1e8):
            print("saved%d"%self.outputCount)
            self.savebuffer()
            for k,v in self.activations.items():
                v.clear()


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


