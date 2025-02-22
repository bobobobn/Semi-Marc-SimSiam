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



from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# 检查点目录
checkpoint_dir = r"checkpoints/simsiamda/"

# 处理每个检查点
for i in range(9):  # 遍历 checkpoint_0000 到 checkpoint_0008
    if i == 1:
        continue
    # 创建 t-SNE 模型
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

    nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=100, normal_size=100, excep_size=100)
    ts_dataset = nonLabelCWRUData.get_test()
    X = ts_dataset.X
    y = ts_dataset.y
    import models.costumed_model as costumed_model

    model = costumed_model.StackedCNNEncoderWithPooling(num_classes=64)

    pretrained_model = os.path.join(checkpoint_dir, f"checkpoint_{i:04d}.pth.tar")
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
    # 保存图像
    plt.savefig(f"tsne_da_{max(0, i-1):04d}.png")
    plt.close()
