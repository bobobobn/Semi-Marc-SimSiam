# -*- coding = utf-8 -*-
# @Time : 2024/12/4 22:43
# @Author : bobobobn
# @File : tsne-siam.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

from data import ssv_data
from sklearn.cluster import KMeans
import torch
import os
import torchvision.transforms as transforms
os.chdir('../')
import DA.data_augmentations

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

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


from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

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
    print(f"Silhouette Coefficient: {silhouette_avg}")

    return cluster_assignments


# 创建 t-SNE 模型
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

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

nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=100, normal_size=100,
                                            excep_size=100)
train_dataset = nonLabelCWRUData.get_test(
    simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation), transforms.Compose(subpolicies)))
ts_dataset = nonLabelCWRUData.get_test(
)
ts_dataset.X = ts_dataset.X[:500]
ts_dataset.y = ts_dataset.y[:500]
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=200, shuffle=True,
    pin_memory=True, drop_last=False)

import models.costumed_model as costumed_model
model = costumed_model.StackedCNNEncoderWithPooling(num_classes=64)

pretrained_model = r"checkpoints\checkpoint_0799.pth.tar"
print("=> loading checkpoint '{}'".format(pretrained_model))
checkpoint = torch.load(pretrained_model, map_location="cpu")

# rename moco pre-trained keys
state_dict = checkpoint['state_dict']
if checkpoint['arch'] != 'fine_tune':
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('encoder.') and not k.startswith('encoder.fc'):
            if k.startswith('encoder.encoder'):
                state_dict[k[len("encoder."):]] = state_dict[k]
            else:
                state_dict[k[len("encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
else:
    for k in list(state_dict.keys()):
        if not k.startswith('encoder.'):
            del state_dict[k]
msg = model.load_state_dict(state_dict, strict=False)
print("missing keys:", set(msg.missing_keys))
# assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model_no_load = costumed_model.StackedCNNEncoderWithPooling(num_classes=64)
model_no_load = model_no_load.to(device)
all_features_0 = []
all_features_1 = []
all_features_0_noload = []
all_features_1_noload = []
for i, (images, _) in enumerate(train_loader):
    images[0].resize_(images[0].size()[0], 1, images[0].size()[1])
    images[1].resize_(images[1].size()[0], 1, images[1].size()[1])
    images[0], images[1] = images[0].float(), images[1].float()

    images[0] = images[0].cuda()
    images[1] = images[1].cuda()

    feature_0 = model.forward_without_fc(images[0])
    feature_1 = model.forward_without_fc(images[1])

    all_features_0.append(feature_0.detach().cpu().numpy())
    all_features_1.append(feature_1.detach().cpu().numpy())

    feature_0_noload = model_no_load.forward_without_fc(images[0])
    feature_1_noload = model_no_load.forward_without_fc(images[1])

    all_features_0_noload.append(feature_0_noload.detach().cpu().numpy())
    all_features_1_noload.append(feature_1_noload.detach().cpu().numpy())
    break

# 合并特征
all_features_0 = np.concatenate(all_features_0, axis=0)
all_features_1 = np.concatenate(all_features_1, axis=0)
all_features_0_noload = np.concatenate(all_features_0_noload, axis=0)
all_features_1_noload = np.concatenate(all_features_1_noload, axis=0)

# 合并两个数据集的特征，用于可视化
all_features = np.concatenate([all_features_0, all_features_1], axis=0)
all_features_noload = np.concatenate([all_features_0_noload, all_features_1_noload], axis=0)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(all_features)

tsne = TSNE(n_components=2, random_state=42)
reduced_features_noload = tsne.fit_transform(all_features_noload)


filtered_indices = np.where(reduced_features[:, 1] > 30)[0]  # 筛选条件

# 2. 获取符合条件的reduced_features中的点
filtered_reduced_features = reduced_features[filtered_indices]

# 3. 获取all_features_noload中对应的点
filtered_all_features_noload = reduced_features_noload[filtered_indices]

# 4. 绘制符合条件的点的散点图
plt.figure(figsize=(8, 6))

# 绘制reduced_features中第二列大于30的点
plt.scatter(filtered_reduced_features[:, 0], filtered_reduced_features[:, 1], color='b', label='Filtered Features', alpha=0.7)

# 绘制all_features_0_noload中对应的点
plt.scatter(filtered_all_features_noload[:, 0], filtered_all_features_noload[:, 1], color='r', label='Corresponding Features (No Load)', alpha=0.7)

# 添加标签和标题
plt.title('Filtered t-SNE Points and Corresponding Features (No Load)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

