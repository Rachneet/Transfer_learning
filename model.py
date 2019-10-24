# my training module
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import csv
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn import metrics
import sys
from torch.utils.data import Dataset
from cnn_train import *

csv.field_size_limit(sys.maxsize)

def run(datapath):

    model = torch.load(datapath + "amazon_model")

    # check params of model
    # for i,param in model.named_parameters():
    #     print(i,param)

    freeze_layers= True

    # freezing all layers
    if freeze_layers:
        for i, param in model.named_parameters():
            param.requires_grad = False

    # freezing layers till conv layer 6
    ct=[]
    for name,child in model.named_children(): # accessing layer names via named_children()
        #print(name,child)
        if "conv6" in ct: # when conv6 is in list, make grad_true for further layers
            for params in child.parameters():
                params.requires_grad = True

        ct.append(name)

    #print("=================================")

    # view the freezed layers
    # for name, child in model.named_children():
    #     for name_2, params in child.named_parameters():
    #         print(name_2, params.requires_grad)

    # train the modified model
    train(1024,20,0.01,model,datapath)



if __name__=="__main__":

    run("/media/rachneet/Arsenal/reddit/char_cnn/transfer_learning/")