#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import simsiam.loader
import simsiam.builder
from torch.optim.lr_scheduler import ExponentialLR

from DA.data_augmentations import *
# from DA.auto_augmentations import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
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
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# simsiam specific configs:
parser.add_argument('--dim', default=256, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')
parser.add_argument('--ssv_size', default=100, type=int,
                    help='ssv_set size (default: 200)')
parser.add_argument('--normal_size', default=100, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=100, type=int,
                    help='excep_size size (default: 200)')

def main():
    args = parser.parse_args()
    args.fix_pred_lr = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    losses, stds, accs = main_worker(args.gpu, ngpus_per_node, args)
    losses_no_sg, stds_no_sg, accs_no_sg = main_worker_no_sg(args.gpu, ngpus_per_node, args)
    with open("train_process.txt", "a") as f:
        f.write(str({"simsiam":{"loss":losses, "std": stds, "accs":accs}}))
        f.write("\n")
        f.write(str({"simsiam_without_sg":{"loss":losses_no_sg, "std": stds_no_sg, "accs":accs_no_sg}}))
        f.write("\n")
    from matplotlib import pyplot as plt
    # 创建一个图形
    plt.figure(figsize=(12, 6))

    # 绘制 Losses 图
    plt.subplot(1, 2, 1)  # (行, 列, 当前位置)
    plt.plot(losses, label='with stop-grad', color='blue', marker='')
    plt.plot(losses_no_sg, label='without stop-grad', color='orange', marker='')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()

    # 绘制 Standard Deviations 图
    plt.subplot(1, 2, 2)  # (行, 列, 当前位置)
    plt.plot(stds, label='with stop-grad', color='blue', marker='')
    plt.plot(stds_no_sg, label='without stop-grad', color='orange', marker='')
    plt.xlabel('Epoch')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation vs Epoch')
    plt.legend()
    plt.show()

def main_worker_no_sg(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    import models.costumed_model
    baseModel = models.costumed_model.StackedCNNEncoderWithPooling
    model = simsiam.builder.SimSiamWOStopGrad(
        baseModel,
        args.dim, args.pred_dim)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model) # print model

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    # initial_lr = 0.05
    # optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Define the exponential decay scheduler
    gamma = 0.95  # Decay factor
    # scheduler = ExponentialLR(optimizer, gamma=gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        AddGaussianNoiseSNR(snr=6),
        TimeShift(512),
        RandomChunkShuffle(30),
        RandomCrop([5], 100),
        RandomScaled((0.5, 1.5)),
    ]
    sec_augmentation = [
        AddGaussianNoiseSNR(snr=6),
        RandomNormalize(),
        PhasePerturbation(0.2),
        RandomChunkShuffle(30),
        RandomCrop([5], 100),
        RandomScaled((0.5, 1.5)),
        RandomAbs(),
        RandomVerticalFlip(),
        RandomReverse(),
    ]
    import DA.auto_augmentations as auto_aug
    policies = [
        auto_aug.SubPolicy(auto_aug.AddGaussianNoiseSNR, scales=(2,6)),
        auto_aug.SubPolicy(auto_aug.RandomNormalize,(0,0.5)),
        auto_aug.SubPolicy(auto_aug.PhasePerturbation, (0.1, 0.5)),
        auto_aug.SubPolicy(auto_aug.RandomChunkShuffle, (10, 100)),
        auto_aug.SubPolicy(auto_aug.RandomCrop,(1,5)),
        auto_aug.SubPolicy(auto_aug.RandomScaled,(0.05,0.6)),
        auto_aug.SubPolicy(auto_aug.RandomAbs),
        auto_aug.SubPolicy(auto_aug.RandomVerticalFlip),
        auto_aug.SubPolicy(auto_aug.RandomReverse),
    ]
    x = [9.76137495,9.62841475,6.13286137,9.50820234,9.89930726,3.44352795,9.68774306,1.34369853,7.87375472,8.19615979,5.77662035,7.82064208,4.02035759,4.39296966,4.42003606]
    subpolicies = []
    idx = 0
    for i in range(len(policies)):
        policy = policies[i]
        p = scale = 1
        if policy.need_p():
            p = x[idx]
            idx += 1
        if policy.need_scale():
            scale = x[idx]
            idx += 1

        subpolicies.append(policies[i % len(policies)].get_entity(scale=scale, p=p))
    import data.ssv_data as ssv_data
    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size)
    train_dataset =nonLabelCWRUData.get_ssv(simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(subpolicies)))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler, drop_last=True)
    losses, stds, accs = [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        loss, std = train(train_loader, model, criterion, optimizer, epoch, args)
        X = nonLabelCWRUData.get_test().X
        X = torch.tensor(X).float()
        X.resize_(X.size()[0], 1, X.size()[1])
        X = X.cuda()
        X = model.encoder(X).to('cpu').detach().numpy()
        acc = kmeans_acc(X, nonLabelCWRUData.get_test().y)
        losses.append(loss)
        stds.append(std)
        accs.append(acc)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoints/simsiamwopred/checkpoint_{:04d}.pth.tar'.format(epoch))

    return losses, stds, accs


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    import models.costumed_model
    baseModel = models.costumed_model.StackedCNNEncoderWithPooling
    model = simsiam.builder.SimSiam(
        baseModel,
        args.dim, args.pred_dim)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model) # print model

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    if args.fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    # initial_lr = 0.05
    # optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Define the exponential decay scheduler
    gamma = 0.95  # Decay factor
    # scheduler = ExponentialLR(optimizer, gamma=gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        AddGaussianNoiseSNR(snr=6),
        TimeShift(512),
        RandomChunkShuffle(30),
        RandomCrop([5], 100),
        RandomScaled((0.5, 1.5)),
    ]
    sec_augmentation = [
        AddGaussianNoiseSNR(snr=6),
        RandomNormalize(),
        PhasePerturbation(0.2),
        RandomChunkShuffle(30),
        RandomCrop([5], 100),
        RandomScaled((0.5, 1.5)),
        RandomAbs(),
        RandomVerticalFlip(),
        RandomReverse(),
    ]
    import DA.auto_augmentations as auto_aug
    policies = [
        auto_aug.SubPolicy(auto_aug.AddGaussianNoiseSNR, scales=(2,6)),
        auto_aug.SubPolicy(auto_aug.RandomNormalize,(0,0.5)),
        auto_aug.SubPolicy(auto_aug.PhasePerturbation, (0.1, 0.5)),
        auto_aug.SubPolicy(auto_aug.RandomChunkShuffle, (10, 100)),
        auto_aug.SubPolicy(auto_aug.RandomCrop,(1,5)),
        auto_aug.SubPolicy(auto_aug.RandomScaled,(0.05,0.6)),
        auto_aug.SubPolicy(auto_aug.RandomAbs),
        auto_aug.SubPolicy(auto_aug.RandomVerticalFlip),
        auto_aug.SubPolicy(auto_aug.RandomReverse),
    ]
    x = [9.76137495,9.62841475,6.13286137,9.50820234,9.89930726,3.44352795,9.68774306,1.34369853,7.87375472,8.19615979,5.77662035,7.82064208,4.02035759,4.39296966,4.42003606]
    subpolicies = []
    idx = 0
    for i in range(len(policies)):
        policy = policies[i]
        p = scale = 1
        if policy.need_p():
            p = x[idx]
            idx += 1
        if policy.need_scale():
            scale = x[idx]
            idx += 1

        subpolicies.append(policies[i % len(policies)].get_entity(scale=scale, p=p))
    import data.ssv_data as ssv_data
    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size)
    train_dataset =nonLabelCWRUData.get_ssv(simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(subpolicies)))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler, drop_last=True)
    losses, stds, accs = [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        loss, std = train(train_loader, model, criterion, optimizer, epoch, args)
        X = nonLabelCWRUData.get_test().X
        X = torch.tensor(X).float()
        X.resize_(X.size()[0], 1, X.size()[1])
        X = X.cuda()
        X = model.encoder(X).to('cpu').detach().numpy()
        acc = kmeans_acc(X, nonLabelCWRUData.get_test().y)
        losses.append(loss)
        stds.append(std)
        accs.append(acc)
    return losses, stds, accs


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    stds = AverageMeter('std', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images[0].resize_(images[0].size()[0], 1, images[0].size()[1])
        images[1].resize_(images[1].size()[0], 1, images[1].size()[1])
        images[0], images[1] = images[0].float(), images[1].float()

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))
        stds.update(compute_l2_std(p1.detach().cpu().numpy()), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg, stds.avg

def kmeans_acc(x, y):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import MinMaxScaler
    def compute_kmeans_acc(y_pred, y_true):
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import accuracy_score
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

        # 使用匈牙利算法找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # 最大化匹配

        # 根据最佳匹配调整预测标签
        mapping = dict(zip(col_ind, row_ind))
        y_pred_mapped = np.array([mapping[label] for label in y_pred])

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred_mapped)
        print(f"Accuracy: {accuracy:.2f}")
        return accuracy

    def get_kmeans_labels(feature):
        # 数据标准化
        scaler = MinMaxScaler()
        normalized_feature = scaler.fit_transform(feature)

        # 设置聚类数量
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=42, verbose=0)

        # 训练 KMeans
        kmeans.fit(normalized_feature)
        cluster_assignments = kmeans.labels_  # 每个样本的聚类标签
        inertia = kmeans.inertia_

        # 计算轮廓系数
        silhouette_avg = silhouette_score(normalized_feature, cluster_assignments)

        return cluster_assignments

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(x)
    labels = get_kmeans_labels(X_tsne)
    kmeans_acc = compute_kmeans_acc(labels, y)
    return kmeans_acc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def compute_l2_std(A):
    # 1. 计算每列的 L2 范数（按列计算，axis=0）
    row_norms = np.linalg.norm(A, axis=1)
    row_norms = row_norms.reshape(-1, 1)
    # 2. 对每列进行 L2 归一化
    A_normalized = A / row_norms

    # 3. 计算每列的标准差（按列计算，axis=0）
    col_std = np.std(A_normalized, axis=0)

    # 4. 计算所有列标准差的均值
    mean_std = np.mean(col_std)
    return mean_std

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    os.chdir('../')
    import sys
    sys.path.append(os.getcwd())
    main()
