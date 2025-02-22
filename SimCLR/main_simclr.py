import sys
sys.path.append('..')
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import simsiam.simsiam.loader
from torch.optim.lr_scheduler import ExponentialLR

from simsiam.DA.data_augmentations import *
import models.costumed_model
# from DA.auto_augmentations import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default=r"C:\Users\bobobob\Desktop\imgnet",
                    help='path to dataset')
parser.add_argument('-dataset-name', default='imagenet',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true', default=False,
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


parser.add_argument('--ssv_size', default=100, type=int,
                    help='ssv_set size (default: 200)')
parser.add_argument('--normal_size', default=100, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=100, type=int,
                    help='excep_size size (default: 200)')

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

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
    import simsiam.DA.auto_augmentations as auto_aug
    policies = [
        auto_aug.SubPolicy(auto_aug.AddGaussianNoiseSNR, scales=(2, 6)),
        auto_aug.SubPolicy(auto_aug.RandomNormalize, (0, 0.5)),
        auto_aug.SubPolicy(auto_aug.PhasePerturbation, (0.1, 0.5)),
        auto_aug.SubPolicy(auto_aug.RandomChunkShuffle, (10, 100)),
        auto_aug.SubPolicy(auto_aug.RandomCrop, (1, 5)),
        auto_aug.SubPolicy(auto_aug.RandomScaled, (0.05, 0.6)),
        auto_aug.SubPolicy(auto_aug.RandomAbs),
        auto_aug.SubPolicy(auto_aug.RandomVerticalFlip),
        auto_aug.SubPolicy(auto_aug.RandomReverse),
    ]
    x = [9.76137495, 9.62841475, 6.13286137, 9.50820234, 9.89930726, 3.44352795, 9.68774306, 1.34369853, 7.87375472,
         8.19615979, 5.77662035, 7.82064208, 4.02035759, 4.39296966, 4.42003606]
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
    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size,
                                                excep_size=args.excep_size)
    train_dataset = nonLabelCWRUData.get_ssv(
        simsiam.simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(subpolicies)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
         pin_memory=True, drop_last=True)
    baseModel = models.costumed_model.StackedCNNEncoderWithPooling
    model = models.resnet_simclr.ResNetSimCLR(base_model=baseModel, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr * args.batch_size / 256, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    import os
    import sys
    sys.path.append('..')
    os.chdir('../')
    sys.path.append(os.getcwd())
    main()
