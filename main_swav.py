# -*- coding = utf-8 -*-
# @Time : 2024/12/25 15:44
# @Author : bobobobn
# @File : main_swav.py
# @Software: PyCharm
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC
from simsiam.DA.data_augmentations import *

from multicropdataset import MultiCropDataset

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument('--pretrained_model', metavar='DIR', help='path to dataset', default=r"C:\Users\bobobob\Desktop\1D-CNN-for-CWRU-master\checkpoints\checkpoint_0500.pth.tar")
parser.add_argument("--data_path", type=str, default=r"C:\Users\bobobob\Desktop\imgnet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=300, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=512, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument('--ssv_size', default=100, type=int,
                    help='ssv_set size (default: 200)')
parser.add_argument('--normal_size', default=100, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=100, type=int,
                    help='excep_size size (default: 200)')

def main():
    global args
    torch.cuda.set_device(0)
    args = parser.parse_args()

    # 构建数据集
    augmentation = [
        AddGaussianNoiseSNR(snr=6),
        # PhasePerturbation(0.2),
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
    import data.ssv_data as ssv_data
    import simsiam.simsiam.loader as siamloader
    import torchvision.transforms as transforms
    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size)
    train_dataset =nonLabelCWRUData.get_ssv(siamloader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(sec_augmentation)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 非分布式需要启用 shuffle
        # num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # 构建模型
    from swav_builder import SwavNet
    import models.costumed_model
    baseModel = models.costumed_model.StackedCNNEncoderWithPooling
    model = SwavNet(baseModel, 128,
                    normalize=True,
                    hidden_mlp=args.hidden_mlp,
                    output_dim=args.feat_dim,
                    nmb_prototypes=args.nmb_prototypes,
                    )
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    model = model.cuda(0)
    logger.info("Building model done.")

    # 构建优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                                                                                           math.cos(math.pi * t / (
                                                                                                       len(train_loader) * (
                                                                                                           args.epochs - args.warmup_epochs))))
                                   for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # 初始化混合精度
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # 加载检查点
    to_restore = {"epoch": 0}

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = True
    print("=> loading checkpoint '{}'".format(args.pretrained_model))
    checkpoint = torch.load(args.pretrained_model, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    msg = model.load_state_dict(state_dict, strict=False)


    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # 训练网络
        scores = train(train_loader, model, optimizer, epoch, lr_schedule)

        # 保存检查点
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if args.use_fp16:
            save_dict["amp"] = apex.amp.state_dict()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='checkpoints/swav/checkpoint_{:04d}.pth.tar'.format(epoch))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(train_loader, model, optimizer, epoch, lr_schedule):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    model.train()

    end = time.time()
    for it, (images, _) in enumerate(train_loader):
        images[0].resize_(images[0].size()[0], 1, images[0].size()[1])
        images[1].resize_(images[1].size()[0], 1, images[1].size()[1])
        images[0], images[1] = images[0].float(), images[1].float()
        images[0] = images[0].cuda(0, non_blocking=True)
        images[1] = images[1].cuda(0, non_blocking=True)
        data_time.update(time.time() - end)

        # 更新学习率
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ 多分辨率前向传播 ... ============
        embedding, output = model(images)
        embedding = embedding.detach()
        bs = images[0].size(0)

        # ============ swav 损失计算 ... ============
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # 获取分配
                q = sinkhorn(out)[-bs:]

            # 聚类分配预测
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)

        # ============ 反向传播和优化步骤 ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ============ 统计信息 ... ============
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if it % args.print_freq == 0:
            progress.display(it)
    return losses.avg


@torch.no_grad()
def sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]

    Q /= torch.sum(Q)
    for it in range(args.sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()

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


if __name__ == "__main__":
    main()
