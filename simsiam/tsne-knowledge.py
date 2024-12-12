# -*- coding = utf-8 -*-
# @Time : 2024/12/5 19:13
# @Author : bobobobn
# @File : tsne-knowledge.py
# @Software: PyCharm
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
import torch
import os
import torchvision.transforms as transforms
os.chdir('../')
import DA.data_augmentations
# 创建 t-SNE 模型
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=100, normal_size=100, excep_size=100)
ts_dataset = nonLabelCWRUData.get_test(
)
X = ts_dataset.X
y = ts_dataset.y
import models.costumed_model as costumed_model
# model = costumed_model.StackedCNNEncoderWithPooling(num_classes=64)
model = costumed_model.CNN_Fine(class_num=6, feature_num=32, fine_tune=True)

pretrained_model = r"C:\Users\bobobob\Desktop\1D-CNN-for-CWRU-master\checkpoints\epoch499_acc0.04441987723112106ckpt.pth.tar"
checkpoint = torch.load(pretrained_model, map_location=torch.device(f'cuda:0'))
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    # if 'linear' not in k and 'fc' not in k:
    new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)
print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
X = torch.tensor(X).float()
X.resize_(X.size()[0], 1, X.size()[1])
X = X.to(device)
X = model.forward_without_fc(X).to('cpu').detach().numpy()
# 降维
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
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