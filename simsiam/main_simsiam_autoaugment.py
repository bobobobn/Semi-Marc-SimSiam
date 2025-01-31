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

from DA.auto_augmentations import *

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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
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
parser.add_argument('-p', '--print-freq', default=30, type=int,
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
parser.add_argument('--normal_size', default=10, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=10, type=int,
                    help='excep_size size (default: 200)')

from types import SimpleNamespace

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

    # policies contains 7*2+3 indices
    policies = [
        SubPolicy(AddGaussianNoiseSNR, scales=(2,6)),
        SubPolicy(RandomNormalize,(0,0.5)),
        SubPolicy(PhasePerturbation, (0.1, 0.5)),
        SubPolicy(RandomChunkShuffle, (10, 100)),
        SubPolicy(RandomCrop,(1,5)),
        SubPolicy(RandomScaled,(0.05,0.6)),
        SubPolicy(RandomAbs),
        SubPolicy(RandomVerticalFlip),
        SubPolicy(RandomReverse),
    ]
    import DA.data_augmentations as fixaug
    augmentation = [
        fixaug.AddGaussianNoiseSNR(snr=6),
        fixaug.TimeShift(512),
        fixaug.RandomChunkShuffle(30),
        fixaug.RandomCrop([5], 100),
        fixaug.RandomScaled((0.5, 1.5)),
    ]
    # from autoaug_models import Controller, RandController
    # from torch import optim
    # controller = RandController(policies)
    # mem_softmaxes = []
    # mem_accuracies = []
    # mem_actions = []
    # losses = []
    # optimizer = optim.Adam(controller.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss().cuda()
    # for epoch in range(500):
    #     softmaxes, subpolicies, actions = controller.predict(size=len(policies))
    #     mem_softmaxes.append(softmaxes)
    #     mem_actions.append(actions)
    #     # Simply call main_worker function
    #     cluster_score = main_worker(args.gpu, ngpus_per_node, args, aug1=augmentation, aug2=subpolicies)
    #     mem_accuracies.append(cluster_score)
    #     print("controller epoch:"+str(epoch) + ",score:" + str(cluster_score))
    #     if len(mem_softmaxes) > 5:
    #         # ricardo: I let some epochs pass, so that the normalization is more robust
    #         total_reward = controller.fit(size=len(policies), actions=mem_actions, scores=mem_accuracies, optimizer=optimizer)
    #         print(f"Episode {epoch + 1}: Total Reward = {total_reward}")
    # print(mem_accuracies)

    import cma
    import numpy as np
    # 问题维度
    dim = 15  # 变量的维度
    # 初始化参数
    initial_solution = np.random.uniform(0, 10, dim)  # 初始解在 [0, 10] 之间
    initial_solution = np.array([10,10,5,10,10,2.5,10,3.3,10,10,10,9,5,5,5])
    sigma = 1  # 初始步长
    options = {
        'popsize': 4 + int(3 * np.log(dim)),  # 种群规模
        'bounds': [0, 10],  # 变量边界
        'maxiter': 5,  # 最大迭代次数
    }

    # 创建CMA-ES优化器
    es = cma.CMAEvolutionStrategy(initial_solution, sigma, options)
    top_5_solutions = []
    # 优化过程
    while not es.stop():
        solutions = es.ask()  # 生成连续解
        fitnesses = []  # 评估适应度
        for x in solutions:
            subpolicies = []
            idx = 0
            for i in range(len(policies)):
                policy = policies[i % len(policies)]
                scale = 0
                p = 10
                if policy.need_p():
                    p=x[idx]
                    idx += 1
                if policy.need_scale():
                    scale = x[idx]
                    idx += 1
                subpolicies.append(policy.get_entity(scale, p))
            cluster_score = main_worker(args.gpu, ngpus_per_node, args, aug1=augmentation,
                                        aug2=subpolicies)
            fitnesses.append(-cluster_score)
        es.tell(solutions, fitnesses)  # 更新CMA-ES参数
        es.logger.add()
        es.disp()
        top_5_solutions.append([es.result.fbest, es.result.xbest])
        top_5_solutions = sorted(top_5_solutions, key=lambda x: x[0])[:5]  # 按目标函数值排序，保留前 5 个

    es.logger.plot()
    # 优化结果
    best_solution = es.result.xbest
    best_fitness = es.result.fbest

    print("\n优化结束:")
    print(f"最优解: {best_solution}")
    print(f"目标函数值: {best_fitness}")


def main_worker(gpu, ngpus_per_node, args, aug1, aug2):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    import models.costumed_model
    import models.Resnet1d as resnet
    # baseModel = models.costumed_model.CNNEncoder
    # baseModel = resnet.resnet18
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
    import data.ssv_data as ssv_data
    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size)
    ssv_dataset =nonLabelCWRUData.get_ssv(simsiam.loader.TwoCropsTransform(transforms.Compose(aug1), transforms.Compose(aug2)))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(ssv_dataset)
    # else:
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        ssv_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler, drop_last=True)
    sum_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        # scheduler.step()
        if abs(epoch - args.epochs) <= 5:
            sum_acc += fintune_model(model, nonLabelCWRUData.get_train(), nonLabelCWRUData.get_test())
    sum_acc /= 5
    return sum_acc
    # todo:
    #  """
    #  1.模型特征提取 features = model.encoder(x_without_DA)
    #  2.tsne降维 features_tsne = tsne(features)
    #  3.Kmeans聚类,评估聚类scores return scores
    #  """
    train_dateset = nonLabelCWRUData.get_train()
    # features = compute_features(train_dateset, model.encoder, 0)
    # labels, _, score = get_kmeans_labels(features, 10)
    # kmeans_acc = compute_kmeans_acc(labels, train_dateset.y)

def fintune_model(model, train_dateset, test_dataset):
    # todo: froze encoder, then train
    from train import train as fintune
    from models import costumed_model
    fine_model = costumed_model.StackedCNNEncoderWithPooling(num_classes=10)
    fine_model.encoder.load_state_dict(model.encoder.encoder.state_dict())
    for name, param in fine_model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    fineargs = SimpleNamespace()
    fineargs.epochs = 20
    fineargs.batch_size = 32
    fineargs.pretrained = False
    fineargs.pretrained_model = ""
    acc = fintune(fine_model, train_dateset, test_dataset, fineargs)
    return acc

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

def compute_features(dataset, model, N):

    X = dataset.X
    X = torch.tensor(X).float()
    X.resize_(X.size()[0], 1, X.size()[1])
    X = X.cuda()
    features = model.forward_without_fc(X).to('cpu').detach().numpy()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    features = tsne.fit_transform(features)
    return features



def get_kmeans_labels(feature, N):
    # 数据标准化
    scaler = MinMaxScaler()
    normalized_feature = scaler.fit_transform(feature)

    # 设置聚类数量
    n_clusters = N
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=42, verbose=0)

    # 训练 KMeans
    kmeans.fit(normalized_feature)
    cluster_assignments = kmeans.labels_  # 每个样本的聚类标签
    inertia = kmeans.inertia_

    # 计算轮廓系数
    silhouette_avg = silhouette_score(normalized_feature, cluster_assignments)
    print(f"Silhouette Coefficient: {silhouette_avg}")

    return cluster_assignments, inertia, silhouette_avg


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
