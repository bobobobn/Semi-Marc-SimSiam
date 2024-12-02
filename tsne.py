# -*- coding = utf-8 -*-
# @Time : 2023/10/25 11:10
# @Author : bobobobn
# @File : tsne.py
# @Software: PyCharm
from time import time
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from sklearn import datasets
from sklearn.manifold import TSNE
from data import create_dataset
from models import create_model
from torch.utils.data import DataLoader


opt = Config()

def nomalize(data):
    data = (data-np.min(data))/(np.max(data) - np.min(data))
    return data

def get_DE():
    tr_dataset = create_dataset(opt.train_dir, train=True)
    tr_loader = DataLoader(tr_dataset, batch_size=opt.batch_size, shuffle=True)
    tr_dataset.X = nomalize(tr_dataset.X)
    return tr_dataset.X,tr_dataset.y

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    num = [0,0,0,0,0,0]
    for lb in label:
        num[lb] += 1
    data_num1 = []
    for k, v in enumerate(label):
        if v == 1:
            data_num1.append(data[k, :])
    data_num1 = data_num1[:20]
    label_num1 = np.ones((len(data_num1)))
    # 删除数字1的数据
    index_to_keep = label!=1
    data = data[index_to_keep]
    label = label[index_to_keep]
    # 合并数字1和其他
    data = np.vstack((data, np.array(data_num1)))
    label = np.hstack((label, label_num1))

    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] ),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plot_tsne(data, label):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    print('result.shape', result.shape)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()

def main():
    data, label, n_samples, n_features = get_data()
    data, label = get_DE()
    print('data.shape',data.shape)
    print('label',label)
    print('label中数字有',len(set(label)),'个不同的数字')
    print('data有',n_samples,'个样本')
    print('每个样本',n_features,'维数据')
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    print('result.shape',result.shape)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()


if __name__ == '__main__':
    main()