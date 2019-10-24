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


model = torch.load("/media/rachneet/Arsenal/reddit/char_cnn/transfer_learning/trained_model_1")

for i,param in model.named_parameters():
    print(i,param)