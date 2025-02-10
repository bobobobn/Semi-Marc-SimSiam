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

import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--pretrained_model', metavar='DIR', help='path to dataset',
                    default=r"checkpoints\simsiam\checkpoint_0799.pth.tar")
parser.add_argument('--pretrained', action='store_true', default=True)
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
        acc = 0
        nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size, beta=beta)
        for i in range(loop):
            model = costumed_model.StackedCNNEncoderWithPooling(num_classes=args.num_classes)
            train(model, nonLabelCWRUData.get_train(), nonLabelCWRUData.get_test(), args)
            from models.marc import Marc
            marc_model = Marc(model, args.num_classes)
            acc += marc(marc_model,nonLabelCWRUData.get_train(), nonLabelCWRUData.get_test(), args)
        acc /= loop
        results.append({"beta":beta, "acc":acc})
        print(results)

    with open("train_results.txt", "a") as f:
        f.write(str(args.pretrained_model) + "marc:" + str(results))
        f.write("\n")

def train(model, tr_dataset, val_dataset, args):
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

    class_counts = torch.zeros(args.num_classes)
    for _, label in tr_dataset:
        class_counts[label] += 1

    from models.costumed_model import ClassBalancedLoss
    loss_fn = ClassBalancedLoss(class_counts)

    ###==============training=================###
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # one epoch
    data_loss = []
    val_acc_list = []
    model = model.to(device)

    for epoch in range(epochs):
        t0 = time.time()
        print('Starting epoch %d / %d' % (epoch + 1, epochs))
        optimizer.step()
        # scheduler.step()
        # set train model or val model for BN and Dropout layers
        model.train()
        # one batch
        for t, (x, y) in enumerate(tr_loader):
            # add one dim to fit the requirements of conv1d layer
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # loss and predictions
            scores = model(x)
            loss = loss_fn(scores, y)
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
        train_acc, _= check_accuracy(model, tr_loader, device)
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

    val_acc, _= check_accuracy(model, val_loader, device)
    return val_acc

def marc(model, tr_dataset, val_dataset, args):
    tr_dataset = data_preprocess.CWRUdata(tr_dataset.X, tr_dataset.y)
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

    class_counts = torch.zeros(args.num_classes)
    for _, label in tr_dataset:
        class_counts[label] += 1

    from models.costumed_model import ClassBalancedLoss
    loss_fn = ClassBalancedLoss(class_counts)

    ###==============training=================###
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # one epoch
    data_loss = []
    val_acc_list = []
    model = model.to(device)

    for epoch in range(epochs):
        t0 = time.time()
        print('Starting epoch %d / %d' % (epoch + 1, epochs))
        optimizer.step()
        # scheduler.step()
        # set train model or val model for BN and Dropout layers
        model.train()
        # one batch
        for t, (x, y) in enumerate(tr_loader):
            # add one dim to fit the requirements of conv1d layer
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # loss and predictions
            scores = model(x)
            loss = loss_fn(scores, y)
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
        train_acc, _= check_accuracy(model, tr_loader, device)
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

# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
# model = model.to(device)
# X = torch.tensor(val_dataset.X).float()
# X.resize_(X.size()[0], 1, X.size()[1])
# X = X.to(device)
# X = model.forward_without_fc(X).to('cpu').detach().numpy()
# # 降维
# X_tsne = tsne.fit_transform(X)
# y = val_dataset.y
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
# plt.colorbar(scatter, label='Classes')
# plt.title("t-SNE Visualization")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.show()
#
# plt.plot(range(len(data_loss)),data_loss)
# plt.xlabel(u'steps')
# plt.ylabel(u'loss')
# plt.show()
#
# plt.plot(range(len(val_acc_list)),val_acc_list)
# plt.xlabel(u'steps')
# plt.ylabel(u'val acc')
# plt.show()
# print('kernel num1: {}'.format(opt.model_param['kernel_num1']))
# print('kernel num2: {}'.format(opt.model_param['kernel_num2']))
# print('Best val Acc: {:4f}'.format(best_acc))
#
# # load best model weights
# model.load_state_dict(best_model_wts)
# val_acc, confuse_matrix = check_accuracy(model, val_loader, device, error_analysis=True)
# # write the confuse_matrix to Excel
# data_pd = pd.DataFrame(confuse_matrix)
# writer = pd.ExcelWriter('results\\confuse_matrix_rate.xlsx')
# data_pd.to_excel(writer)
# writer.save()
# writer.close()
# # save model in results dir
# model_save_path = 'results\\' + time.strftime('%Y%m%d%H%M_') + str(int(100*best_acc)) + '.pth'
# torch.save(model.state_dict(), model_save_path)
# print('best model is saved in: ', model_save_path)

