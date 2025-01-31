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

parser = argparse.ArgumentParser(description='Apply standard trained model to generate labels on unlabeled data')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
# load trained models
parser.add_argument('--resume', type=str, default='')
# data related
parser.add_argument('--output_dir', default='./data', type=str)
parser.add_argument('--output_filename', default='pseudo_labeled_cwru.pth', type=str)
parser.add_argument('--pretrained_model', metavar='DIR', help='path to dataset',
                    default=r"checkpoints\finetune\checkpoint_0019.pth.tar")
parser.add_argument('--ssv_size', default=100, type=int,
                    help='ssv_set size (default: 200)')
parser.add_argument('--normal_size', default=100, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=10, type=int,
                    help='excep_size size (default: 200)')

args = parser.parse_args()

model = costumed_model.StackedCNNEncoderWithPooling(num_classes=10)

if os.path.isfile(args.pretrained_model):
    checkpoint = torch.load(args.pretrained_model, map_location="cpu")
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    assert len(set(msg.missing_keys)) == 0
else:
    raise ValueError(f"No checkpoint found at '{args.pretrained_model}'")

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
else:
    model = torch.nn.DataParallel(model).cuda()


cudnn.benchmark = True

model.eval()
nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size)
ssv_dataset = nonLabelCWRUData.get_ssv()
X = ssv_dataset.X
X = torch.tensor(X).float()
X.resize_(X.size()[0], 1, X.size()[1])
X = X.cuda()
y = model(X)
y = torch.argmax(y, dim=1)
y = y.to("cpu").detach().numpy()
ssv_dataset.y = y
out_path = os.path.join(args.output_dir, args.output_filename)
torch.save(ssv_dataset, out_path)

dataset = torch.load(out_path)
print(dataset)
