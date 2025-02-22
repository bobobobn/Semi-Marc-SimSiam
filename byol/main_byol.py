# -*- coding = utf-8 -*-
# @Time : 2025/2/15 15:04
# @Author : bobobobn
# @File : main_byol.py
# @Software: PyCharm
import sys
sys.path.append('..')
import torch
from byol_pytorch import BYOL
from torchvision import models
import argparse
import torch.backends.cudnn as cudnn
from simsiam.DA.data_augmentations import *
import simsiam.DA.auto_augmentations as auto_aug
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
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
parser.add_argument('--num_classes', default=10, type=int)


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA is available')
    import models.costumed_model
    model = models.costumed_model.StackedCNNEncoderWithPooling(args.num_classes).cuda()
    main_worker(model, args)


def main_worker(model, args):
    augmentation = [
        AddGaussianNoiseSNR(snr=6),
        TimeShift(512),
        RandomChunkShuffle(30),
        RandomCrop([5], 100),
        RandomScaled((0.5, 1.5)),
    ]
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
    learner = BYOL(
        model,
        image_size=1024,
        hidden_layer=-1,
        augment_fn=transforms.Compose(augmentation),
        augment_fn2=transforms.Compose(subpolicies)
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    import data.ssv_data as ssv_data
    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size,
                                                excep_size=args.excep_size)
    train_dataset = nonLabelCWRUData.get_ssv()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False)
    for epoch in range(args.epochs):
        train(train_loader, learner, opt, epoch, args)

    # save your improved network
    save_checkpoint({
                'epoch': args.epochs + 1,
                'arch': "fine_tune",
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
            }, is_best=False,
                filename='checkpoints/byol/checkpoint_{:04d}_batchsize_{:04d}.pth.tar'.format(epoch, args.batch_size))


def train(train_loader, learner, opt, epoch, args):
    for t, (x, y) in enumerate(train_loader):
        images = x.cuda()
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        print("epoch:", epoch, ", loss:", loss)
        opt.step()
        learner.update_moving_average()  # update moving average of target encoder


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    import shutil
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == "__main__":
    import os
    os.chdir('../')
    import sys
    sys.path.append(os.getcwd())
    main()