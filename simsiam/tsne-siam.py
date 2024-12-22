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


def get_kmeans_labels(feature):
    # 创建 KMeans 对象
    n_clusters = 10  # 聚类数量
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=42, verbose=1)

    # 训练 KMeans
    kmeans.fit(feature)
    cluster_assignments = kmeans.labels_  # 每个样本的聚类标签
    return cluster_assignments


# 创建 t-SNE 模型
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=100, normal_size=100, excep_size=100)
ts_dataset = nonLabelCWRUData.get_test(
)
X = ts_dataset.X
# for i in range(len(X)):
#     X[i] = transforms.Compose([DA.data_augmentations.GaussianWeightedMovingAverage(10, 1)])(X[i])
y = ts_dataset.y
for i in range(len(y)):
    l = y[i]
    if l > 9:
        y[i] = 10
X = X[y != 10]
y = y[y != 10]
import models.Resnet1d as resnet
# model = resnet.resnet18NOFc(num_classes=6)
import models.costumed_model as costumed_model
model = costumed_model.StackedCNNEncoderWithPooling(num_classes=64)

pretrained_model = r"C:\Users\bobobob\Desktop\1D-CNN-for-CWRU-master\checkpoints\checkpoint_0579.pth.tar"
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = True
print("=> loading checkpoint '{}'".format(pretrained_model))
checkpoint = torch.load(pretrained_model, map_location="cpu")

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
# assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X = torch.tensor(X).float()
X.resize_(X.size()[0], 1, X.size()[1])
X = X.to(device)
X = model.forward_without_fc(X).to('cpu').detach().numpy()
# 降维
X_tsne = tsne.fit_transform(X)

labels = get_kmeans_labels(X_tsne)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=10)
plt.colorbar(scatter, label='Classes')
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# X = ts_dataset.X
# # 从每个类别中选取一个样本
# unique_labels = np.unique(y)
# selected_signals = []
# selected_labels = []
#
# for label in unique_labels:
#     # 按标签筛选，选第一个匹配的样本
#     idx = np.where(y == label)[0][0]
#     selected_signals.append(X[idx])
#     selected_labels.append(label)
#
# # 绘制选中的信号
# plt.figure(figsize=(12, 6))
# for i, (signal, label) in enumerate(zip(selected_signals, selected_labels)):
#     plt.subplot(1, len(unique_labels), i + 1)
#     plt.plot(signal)
#     plt.title(f"Label: {label}")
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.grid(True)
#
# plt.tight_layout()
# plt.show()


