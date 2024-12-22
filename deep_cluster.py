# -*- coding = utf-8 -*-
# @Time : 2024/12/21 18:43
# @Author : bobobobn
# @File : deep_cluster.py.py
# @Software: PyCharm

import clustering
import argparse
import os
import pickle
import time

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering
import models
from utils import AverageMeter, Logger, UnifLabelSampler


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--pretrained_model', metavar='DIR', help='path to dataset', default=r"C:\Users\bobobob\Desktop\1D-CNN-for-CWRU-master\checkpoints\checkpoint_0500.pth.tar")
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='checkpoints', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--tsne', action='store_true', help='chatty')

    parser.add_argument('--ssv_size', default=100, type=int,
                        help='ssv_set size (default: 200)')
    parser.add_argument('--normal_size', default=100, type=int,
                        help='normal_size size (default: 200)')
    parser.add_argument('--excep_size', default=100, type=int,
                        help='excep_size size (default: 200)')
    return parser.parse_args()


def main(args):
    # fix random seeds
    args.verbose = True
    # args.tsne = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    import models.costumed_model as costumed_model
    model = costumed_model.StackedCNNEncoderWithPooling(num_classes=10)
    fd = int(model.num_classes)
    # model.fc = None
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10**args.wd,
    )

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = True
    print("=> loading checkpoint '{}'".format(args.pretrained_model))
    checkpoint = torch.load(args.pretrained_model, map_location="cpu")

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

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # creating cluster assignments log
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # load the data
    end = time.time()

    import data.ssv_data as ssv_data
    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=args.ssv_size, normal_size=args.normal_size, excep_size=args.excep_size)
    dataset = nonLabelCWRUData.get_ssv()

    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](fd)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        # remove head
        model.top_layer = None
        # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # get the features for the whole dataset
        features = compute_features(dataset, model, len(dataset))

        # cluster the features
        if args.verbose:
            print('Cluster the features')
        kmeans_labels, cluster_loss = get_kmeans_labels(features)
        # assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        dataset.y = kmeans_labels
        # uniformly sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(dataset)),
                                   dataset.y)


        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.fc.children())
        mlp.append(nn.ReLU(inplace=True).cuda())
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(dataset.y))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.cuda()

        # train network with clusters as pseudo-labels
        end = time.time()
        loss = train(train_dataloader, model, criterion, optimizer, epoch)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, cluster_loss, loss))
            # try:
            #     nmi = normalized_mutual_info_score(
            #         clustering.arrange_clustering(dataset),
            #         clustering.arrange_clustering(cluster_log.data[-1])
            #     )
            #     print('NMI against previous assignment: {0:.3f}'.format(nmi))
            # except IndexError:
            #     pass
            print('####################### \n')

        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))


        if args.tsne and epoch % 10 == 0:
            create_tsne(features, kmeans_labels)
        # save cluster assignments
        # cluster_log.log(deepcluster.images_lists)


def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10**args.wd,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        target = target.cuda().long()
        input_var = torch.autograd.Variable(input_tensor.cuda()).float()
        target_var = torch.autograd.Variable(target)
        input_var.resize_(input_var.size()[0], 1, input_var.size()[1])

        output = model(input_var)
        loss = crit(output, target_var)

        # record loss
        losses.update(loss.data, input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg

def compute_features(dataset, model, N):
    X = dataset.X
    X = torch.tensor(X).float()
    X.resize_(X.size()[0], 1, X.size()[1])
    X = X.to(device)
    features = model.forward_without_fc(X).to('cpu').detach().numpy()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    features = tsne.fit_transform(features)
    return features


def create_tsne(features, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter, label='Classes')
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


def get_kmeans_labels(feature):
    # 创建 KMeans 对象
    n_clusters = 10  # 聚类数量
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=42, verbose=1)

    # 训练 KMeans
    kmeans.fit(feature)
    cluster_assignments = kmeans.labels_  # 每个样本的聚类标签
    inertia = kmeans.inertia_
    return cluster_assignments, inertia


if __name__ == '__main__':
    args = parse_args()
    main(args)
