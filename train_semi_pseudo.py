# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:31:08 2018

@author: rlk
"""

from config import Config
from data import create_dataset
from models import create_model
from torch.utils.data import DataLoader
from utils import check_accuracy, check_semi_accuracy
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
import gModel
import dModel
from matplotlib import pyplot as plt
from tsne import plot_tsne
from data import data_preprocess
from models import costumed_model
from data import ssv_data
import math
import random

import torch.backends.cudnn as cudnn
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--pretrained_model', metavar='DIR', help='path to dataset',
                    default=r"checkpoints\simsiam\checkpoint_0799.pth.tar")
parser.add_argument('--output_dir', default='./data', type=str)
parser.add_argument('--output_filename', default='pseudo_labeled_cwru.pth', type=str)
parser.add_argument('--pretrained', action='store_true', default=True)
parser.add_argument('--requires_grad', action='store_true', default=False)
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
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--ssv_size', default=100, type=int,
                    help='ssv_set size (default: 200)')
parser.add_argument('--normal_size', default=100, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=10, type=int,
                    help='excep_size size (default: 200)')
parser.add_argument('--omega', default=1.0, type=float,
                    help='weight of non labeled data')
parser.add_argument('--num_classes', default=10, type=int)

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
    betas = [50, 100]
    results = []
    loop = 1
    for beta in betas:
        nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, beta=beta)
        # out_path = os.path.join(args.output_dir, args.output_filename)
        # ssv_dataset = torch.load(out_path)
        '''
            1.用labeled dataset微调model
            2.用model预测unlabeled dataset
            3.用omega(unlabeled dataset) & (labeled dataset)微调model
        '''
        acc = 0
        for i in range(loop):
            model = costumed_model.StackedCNNEncoderWithPooling(num_classes=10)
            # from train import train as fine_tune
            # fine_tune(model, nonLabelCWRUData.get_train(), nonLabelCWRUData.get_test(), args)
            # from gen_pseudo_labels import gen_pseudo_labels
            # ssv_dataset = gen_pseudo_labels(model, nonLabelCWRUData.get_ssv())
            # semiCWRU = data_preprocess.SemiSupervisedImbalanceCWRU(nonLabelCWRUData.get_train(), ssv_dataset,
            #                                                        omega=args.omega)
            acc += train_semi(model, nonLabelCWRUData.get_train(), nonLabelCWRUData.get_test(), args, nonLabelCWRUData.get_ssv())
        acc /= loop
        results.append({"beta":beta, "acc":acc})
        print(results)
    with open("train_results.txt", "a") as f:
        f.write(str(args.pretrained_model) + "_semi_pseudo:" + str(results))
        f.write("\n")


def train_semi(model, tr_dataset, val_dataset, args, ssv_dataset):
    epochs = args.epochs
    pretrained = args.pretrained
    pretrained_model = args.pretrained_model
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    print('#training num = %d' % len(tr_dataset))

    writer = SummaryWriter(comment=str(opt.model_param['kernel_num1'])+'_'+
                           str(opt.model_param['kernel_num2']))

    total_steps = 0
    from torch.optim.lr_scheduler import ExponentialLR

    init_lr = 0.05 * opt.batch_size / 256
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=init_lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    # Define the exponential decay scheduler
    gamma = 0.95  # Decay factor
    scheduler = ExponentialLR(optimizer, gamma=gamma)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    ###==============training=================###
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # one epoch
    data_loss = []
    val_acc_list = []

    if pretrained:
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = args.requires_grad
        print("=> loading checkpoint '{}'".format(pretrained_model))
        checkpoint = torch.load(pretrained_model, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('encoder') and not k.startswith('encoder.fc'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(set(msg.missing_keys))
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model = model.to(device)
    T1 = 150
    T2 = 250
    for epoch in range(epochs):
        if epoch >= T1:
            from gen_pseudo_labels import gen_pseudo_labels
            ssv_dataset = gen_pseudo_labels(model, ssv_dataset)
            tr_loader = DataLoader(data_preprocess.SemiSupervisedImbalanceCWRU(tr_dataset, ssv_dataset,
                                                                   omega=args.omega*min(1.0, (epoch-T1)/(T2-T1))), batch_size=args.batch_size, shuffle=True)
        t0 = time.time()
        print('Starting epoch %d / %d' % (epoch + 1, epochs))
        optimizer.step()
        # scheduler.step()
        # set train model or val model for BN and Dropout layers
        model.train()
        # one batch
        if epoch < T1:
            loss_fn = torch.nn.CrossEntropyLoss()
            for t, (x, y) in enumerate(tr_loader):
                # add one dim to fit the requirements of conv1d layer
                x.resize_(x.size()[0], 1, x.size()[1])
                x, y = x.float(), y.long()
                x, y = x.to(device), y.to(device)
                # loss and predictions
                scores = model(x)
                loss = loss_fn(scores, y)
                data_loss.append(loss.to("cpu").detach().numpy())
                # print and save loss per 'print_every' times
                if (t + 1) % opt.print_every == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.item()))
                # parameters update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            for t, (x, y, omega) in enumerate(tr_loader):
                # add one dim to fit the requirements of conv1d layer
                x.resize_(x.size()[0], 1, x.size()[1])
                x, y, omega = x.float(), y.long(), omega.float()
                x, y, omega = x.to(device), y.to(device), omega.to(device)
                # loss and predictions
                scores = model(x)
                # 按样本权重计算加权损失
                raw_loss = loss_fn(scores, y)  # 每个样本的损失
                loss = (raw_loss @ omega) / len(raw_loss) # 加权平均损失
                data_loss.append(loss.to("cpu").detach().numpy())
                writer.add_scalar('loss', loss.item())
                # print and save loss per 'print_every' times
                if (t + 1) % opt.print_every == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.item()))
                # parameters update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        adjust_learning_rate(optimizer, init_lr, epoch, epochs)
        # save epoch loss and acc to train or val history
        if epoch < T1:
            train_acc, _ = check_accuracy(model, tr_loader, device)
        else:
            train_acc, _= check_semi_accuracy(model, tr_loader, device)
        val_acc, _= check_accuracy(model, val_loader, device)
        val_acc_list.append(val_acc)
        # writer acc and weight to tensorboard
        writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        t1 = time.time()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='checkpoints/semi/checkpoint_{:04d}.pth.tar'.format(epoch))

    val_acc, _= check_accuracy(model, val_loader, device)
    return val_acc




def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    import shutil
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == "__main__":
    main()