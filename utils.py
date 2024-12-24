# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:38:22 2018

@author: lenovo
"""
import numpy as np
import torch
import pandas as pd

###=====check the acc of model on loader, if error_analysis return confuseMatrix====
def check_accuracy(model, loader, device, error_analysis=False):
    # save the errors samples predicted by model
    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None
    # correct counts
    num_correct = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        # one batch
        for x, y in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # predictions
            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            # accumulate the corrects
            num_correct += preds.eq(y.view_as(preds)).sum().item()
            # confuse matrix: labels and preds
            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))
    acc = float(num_correct) / len(loader.dataset)
    # confuse matrix 
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)
    print('Got %d / %d correct (%.2f)' % (num_correct, len(loader.dataset), 100 * acc))
    return acc, confuse_matrix


# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import models


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformly across pseudolabels.
    Args:
        N (int): size of returned iterator.
        dataset_y (list): list of labels for the dataset.
    """

    def __init__(self, N, dataset_y):
        self.N = N
        self.dataset_y = dataset_y
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        # 获取所有唯一的标签
        unique_labels = np.unique(self.dataset_y)
        nmb_non_empty_clusters = len(unique_labels)

        # 每个标签的样本数量
        size_per_label = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for label in unique_labels:
            # 获取当前标签的所有索引
            label_indexes = np.where(self.dataset_y == label)[0]

            # 采样
            if len(label_indexes) > 0:
                sampled_indexes = np.random.choice(
                    label_indexes,
                    size_per_label,
                    replace=(len(label_indexes) <= size_per_label)
                )
                res = np.concatenate((res, sampled_indexes))

        # 随机化结果
        np.random.shuffle(res)
        res = list(res.astype('int'))

        # 保证返回的索引数量等于 N
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    from clustering import Kmeans
    import clustering
    import numpy as np

    # 初始化 Kmeans
    n_clusters = 10
    kmeans = Kmeans(n_clusters)

    # 聚类特征数据
    features = np.random.rand(1000, 512)  # 假设有 1000 个样本，每个样本是 128 维特征
    cluster_loss = kmeans.cluster(features, verbose=True)

    # 获取每个簇的样本列表
    images_lists = kmeans.images_lists

    images_lists[9] = []
    dataset = \
        clustering.cluster_assign(images_lists,
                                  features)
    # 检测空簇
    empty_clusters = [i for i, cluster in enumerate(images_lists) if len(cluster) == 0]
    print(f"Empty clusters: {empty_clusters}")
    dataset
    # 如果存在空簇，重新分配样本
    if empty_clusters:
        def redistribute_empty_clusters(images_lists):
            largest_cluster_idx = max(range(len(images_lists)), key=lambda i: len(images_lists[i]))
            largest_cluster = images_lists[largest_cluster_idx]

            for i, cluster in enumerate(images_lists):
                if len(cluster) == 0:
                    cluster.append(largest_cluster.pop())
            return images_lists


        images_lists = redistribute_empty_clusters(images_lists)

    print(f"Re-assigned clusters: {[len(cluster) for cluster in images_lists]}")
