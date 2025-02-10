import torch.backends.cudnn as cudnn

import logging
import os
import pickle
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, SVHN
from torchvision import transforms
from models import costumed_model
import data.ssv_data as ssv_data

def gen_pseudo_labels(model, ssv_dataset):
    model.eval()
    X = ssv_dataset.X
    X = torch.tensor(X).float()
    X.resize_(X.size()[0], 1, X.size()[1])
    X = X.cuda()
    y = model(X)
    y = torch.argmax(y, dim=1)
    y = y.to("cpu").detach().numpy()
    ssv_dataset.y = y
    # out_path = os.path.join(args.output_dir, args.output_filename)
    # torch.save(ssv_dataset, out_path)
    #
    # dataset = torch.load(out_path)
    # print(dataset)
    return ssv_dataset
