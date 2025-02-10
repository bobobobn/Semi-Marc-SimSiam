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

nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=100, normal_size=100, excep_size=100)
ts_dataset = nonLabelCWRUData.get_test(
)
X = ts_dataset.X
# for i in range(len(X)):
#     X[i] = transforms.Compose([DA.data_augmentations.GaussianWeightedMovingAverage(10, 1)])(X[i])
y = ts_dataset.y
import models.Resnet1d as resnet
# model = resnet.resnet18NOFc(num_classes=6)
import models.costumed_model as costumed_model
model = costumed_model.StackedCNNEncoderWithPooling(num_classes=64)

pretrained_model = r"checkpoints/simsiamda/checkpoint_0007.pth.tar"
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

labels = get_kmeans_labels(X_tsne)
kmeans_acc = compute_kmeans_acc(labels, y)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
plt.colorbar(scatter, label='Classes')
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()
