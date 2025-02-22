# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:31:08 2018

@author: rlk
"""

from config import Config
from data import create_dataset
from models import create_model
from torch.utils.data import DataLoader
from utils import check_accuracy
import torch
from tensorboardX import SummaryWriter
import copy
import time
import pandas as pd
from models import MLP
from models import Resnet1d
from models import Alexnet1d
from models import BiLSTM1d
from models import LeNet1d
from torchsummary import summary
import numpy as np
opt = Config()
from matplotlib import pyplot as plt
from data import data_preprocess
from models import costumed_model
from data import ssv_data
import math
import random

import torch.backends.cudnn as cudnn

import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--pretrained_model', metavar='DIR', help='path to dataset',
                    default=r"checkpoints\simsiam\checkpoint_0799.pth.tar")
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--ssv_size', default=100, type=int,
                    help='ssv_set size (default: 200)')
parser.add_argument('--normal_size', default=100, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=10, type=int,
                    help='excep_size size (default: 200)')
parser.add_argument('--num_classes', default=10, type=int,
                    help='excep_size size (default: 200)')

import data.ssv_data as ssv_data
import torchvision.transforms as transforms

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA is available')
    betas = [1, 2, 5, 10, 50, 100]
    results = []
    loop = 5
    for beta in betas:
        nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size, beta=beta)
        from utils import compute_time_domain_features
        train_data, test_data = nonLabelCWRUData.get_train(), nonLabelCWRUData.get_test()
        feature_train = compute_time_domain_features(train_data.X)
        feature_test = compute_time_domain_features(test_data.X)
        feature_max = feature_train.max()
        feature_min = feature_train.min()
        feature_train = (feature_train- feature_min) / (feature_max - feature_min)
        feature_test = (feature_test- feature_min) / (feature_max - feature_min)
        from sklearn.metrics import accuracy_score
        from sklearn.svm import SVC
        # 训练SVM分类器
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(feature_train, train_data.y)

        # 预测
        y_pred = svm.predict(feature_test)

        # 计算准确率
        accuracy = accuracy_score(test_data.y, y_pred)
        results.append({"beta":beta, "acc":accuracy})
        print(results)

if __name__ == "__main__":
    main()
