# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:30:05 2018
将CWRU原始数据分为驱动端与风扇端（DE，FE），根据转速与故障将数据分为101中类别，
其中训练样本80%，测试样本20%，随机打乱取样。
最后数据以h5格式存储，其中DE为驱动端测点的数据集，包含训练样本与测试样本；FE为风扇端。
@author: rlk
"""

import h5py
import pandas as pd
import numpy as np
import os
import scipy.io as scio
import torch.utils.data as data
from Ae1d import create_ae_label
from DCAe1d import create_dcae_label

class CWRUdata(data.Dataset):

    def __init__(self, x_set, y_set):
        self.X = x_set
        self.y = y_set

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


'''以下5个函数构造长尾分布数据集用的，直接看create_cwru_dataset()'''


def normalize(data, test_data):
    return (data-np.min(data))/(np.max(data)-np.min(data)), (test_data-np.min(data))/(np.max(data)-np.min(data))


def standardize(data, test_data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    mean = mean.reshape((1, -1))
    std_dev = std_dev.reshape((1, -1))
    return (data - mean)/std_dev, (test_data - mean) / std_dev


def pareto(x,alpha,m=1):
    return alpha*m**alpha/x**(alpha+1)


def pareto_inv(y,alpha,m=1):
    return (alpha*m**alpha/y)**(1/(alpha+1))


def get_pareto_list(head,tail,alpha):
    head_x = pareto_inv(head,alpha,tail)
    tail_x = pareto_inv(tail,alpha,tail)
    x_list = np.arange(head_x,tail_x,(tail_x-head_x)/6.0)
    y_list = pareto(x_list,alpha,tail)
    y_list[0] = head
    y_list[-1] = tail
    return y_list


def train_set_split_ssv(train_x, train_y, ssv_num):
    def split_dataset(X, Y, num_samples_per_class, random_state=42):
        if random_state is not None:
            np.random.seed(random_state)
        test_indices = []
        train_indices = []
        for label, num_samples in enumerate(num_samples_per_class):
            class_indices = np.where(Y == label)[0]
            # 随机选取指定数量的样本索引作为测试集
            test_indices.extend(np.random.choice(class_indices, size=num_samples, replace=False))
            # 剩余的作为训练集
            train_indices.extend([idx for idx in class_indices if idx not in test_indices])
        return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]
    num_samples_per_class = ssv_num * np.ones((6), dtype=int)

    X_train, X_test, Y_train, Y_test = split_dataset(train_x, train_y, num_samples_per_class)
    return X_train, Y_train, X_test


def get_imb_data(signals, labels, excep_num, normal_num):
    def split_dataset(X, Y, num_samples_per_class, random_state=42):
        if random_state is not None:
            np.random.seed(random_state)

        test_indices = []
        train_indices = []
        for label, num_samples in enumerate(num_samples_per_class):
            class_indices = np.where(Y == label)[0]
            # 随机选取指定数量的样本索引作为测试集
            test_indices.extend(np.random.choice(class_indices, size=num_samples, replace=False))
            # 剩余的作为训练集
            train_indices.extend([idx for idx in class_indices if idx not in test_indices])
        return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

    num_samples_per_class = excep_num * np.ones((6), dtype=int)
    num_samples_per_class[0] = normal_num

    X_train, X_test, Y_train, Y_test = split_dataset(signals, labels, num_samples_per_class)
    return X_test, Y_test


def create_knowledge_label(ssv_x):
    feature = []
    # 均值
    mean_val = np.mean(ssv_x, axis=1)
    mean_val = mean_val.reshape((-1, 1))
    feature.append(mean_val)
    # 标准差
    std = np.std(ssv_x, axis=1)
    std = std.reshape((-1, 1))
    feature.append(std)
    # 按行计算平方根的均值
    sqrt_arr = np.sqrt(np.abs(ssv_x))
    row_sqrt_mean = np.mean(sqrt_arr, axis=1)
    row_sqrt_mean = row_sqrt_mean.reshape((-1, 1))
    row_sqrt_mean *= row_sqrt_mean
    feature.append(row_sqrt_mean)
    # 按行计算绝对值的均值
    abs_arr = np.abs(ssv_x)
    row_abs_mean = np.mean(abs_arr, axis=1)
    row_abs_mean = row_abs_mean.reshape((-1, 1))
    feature.append(row_abs_mean)
    # 按行计算三次方的均值
    skewness = ssv_x ** 3
    row_skewness_mean = np.mean(skewness, axis=1)
    row_skewness_mean = row_skewness_mean.reshape((-1, 1))
    feature.append(row_skewness_mean)
    # 按行计算四次方的均值
    kurtosis = ssv_x ** 4
    row_kurtosis_mean = np.mean(kurtosis, axis=1)
    row_kurtosis_mean = row_kurtosis_mean.reshape((-1, 1))
    feature.append(row_kurtosis_mean)
    # 按行计算二次方的均值
    variance = ssv_x ** 2
    row_variance_mean = np.mean(variance, axis=1)
    row_variance_mean = row_variance_mean.reshape((-1, 1))
    feature.append(row_variance_mean)
    # 计算kurtosis index
    kurtosis_index = row_kurtosis_mean / row_variance_mean**2
    feature.append(kurtosis_index)
    # 计算peak index
    peak_index = np.max(abs_arr) / std
    feature.append(peak_index)
    # 计算waveform index
    waveform_index = std / row_abs_mean
    feature.append(waveform_index)
    # 计算pulse index
    pulse_index = np.max(abs_arr) / row_abs_mean
    feature.append(pulse_index)
    # 计算skewness index
    skewness_index = row_skewness_mean / np.sqrt(row_variance_mean)**3
    feature.append(skewness_index)
    # # 频域
    # ssv_fft = np.fft.fft(ssv_x)
    #
    # # 均值
    # mean_val_fft = np.mean(ssv_fft, axis=1)
    # mean_val_fft = mean_val_fft.reshape((-1, 1))
    # feature.append(mean_val_fft)
    # # 方差
    # std_fft = np.std(ssv_fft, axis=1)
    # var_fft = std_fft.reshape((-1, 1))
    # feature.append(var_fft)
    # # fre skewness
    # skew_fft = (ssv_fft - mean_val_fft) ** 3 / (var_fft ** 1.5)
    # skew_fft = np.mean(skew_fft, axis=1)
    # skew_fft = skew_fft.reshape((-1, 1))
    # feature.append(skew_fft)
    # # fre steepness
    # steep_fft = (ssv_fft - mean_val_fft) ** 4 / (var_fft ** 2)
    # steep_fft = np.mean(steep_fft, axis=1)
    # steep_fft = steep_fft.reshape((-1, 1))
    # feature.append(steep_fft)
    # # fre gravity
    # fre_val = np.array(range(ssv_fft.shape[-1])) / ssv_fft.shape[-1]
    # fre_val = fre_val.reshape(1, 1024)
    # grav = fre_val * ssv_fft
    # grav = np.mean(grav, axis=1) / np.mean(ssv_fft, axis=1)
    # grav = grav.reshape((-1,1))
    # feature.append(grav)
    # # fre std
    # diff = (fre_val - grav)**2*ssv_fft / ssv_fft.shape[-1]
    # fre_std = np.sqrt(np.mean(diff, axis=1) / np.mean(ssv_fft, axis=1))
    # fre_std = fre_std.reshape((-1,1))
    # feature.append(fre_std)
    # # fre root mean square
    # root = fre_val**2*ssv_fft
    # root_mean_s = np.sqrt(np.mean(root, axis=1) / np.mean(ssv_fft, axis=1))
    # root_mean_s = root_mean_s.reshape((-1,1))
    # feature.append(root_mean_s)
    # # fre average
    # average_fre = np.sqrt(np.mean(fre_val**4*ssv_fft, axis=1) / np.mean(fre_val**2*ssv_fft, axis=1))
    # average_fre = average_fre.reshape((-1,1))
    # feature.append(average_fre)
    # # regularity degree
    # regularity_degree = np.mean(fre_val**2*ssv_fft, axis=1) * ssv_fft.shape[-1] / np.sqrt(np.mean(ssv_fft, axis=1)/np.mean(fre_val**4*ssv_fft, axis=1))
    # regularity_degree = regularity_degree.reshape((-1,1))
    # feature.append(regularity_degree)
    # # variation para
    # variation_para = fre_std / grav
    # feature.append(variation_para)
    # # eighth-order moment
    # eighth_moment = np.mean((fre_val-grav)**3 * ssv_fft, axis=1)
    # eighth_moment = eighth_moment.reshape((-1,1))
    # eighth_moment /= fre_std ** 3
    # feature.append(eighth_moment)
    # # sixteenth-order moment
    # sixteenth_order = np.mean((fre_val-grav)**4 * ssv_fft, axis=1)
    # sixteenth_order = sixteenth_order.reshape((-1,1))
    # sixteenth_order /= fre_std**4
    # feature.append(sixteenth_order)

    feature = np.hstack(feature)
    feature = np.real(feature)
    mean = np.mean(feature, axis=0)
    std_dev = np.std(feature, axis=0)
    mean = mean.reshape((1,-1))
    std_dev = std_dev.reshape((1,-1))
    # 标准化矩阵
    normalized_matrix = (feature - mean) / std_dev
    normalized_matrix = np.real(normalized_matrix)
    return normalized_matrix

def create_ssv_y_fake_sample(ssv_x, split_len):
    split_len = int(split_len)
    x_fake = np.hstack((ssv_x[:, :split_len], ssv_x[:, split_len:]))
    x = np.vstack((x_fake, ssv_x))
    y = np.vstack((np.ones((len(ssv_x), 1)), np.zeros((len(ssv_x), 1))))
    return x,y




def get_lt_data(signals, labels, imbalance_factor=10, alpha=5):
    label_num = [0,0,0,0,0,0]
    imb_num_list = np.random.pareto(alpha, len(label_num))
    imb_num_list = sorted(imb_num_list)
    imb_num_list = normalize(imb_num_list)
    for label in labels:
        label_num[label] += 1
    label_num = np.array(label_num)
    max_num = max(label_num)
    min_num = min(label_num)
    tail_num = max_num/imbalance_factor
    # lt_label_num = np.round(max_num - ((max_num - tail_num)/max(sorted_indices)) * (max(sorted_indices)-np.array(sorted_indices)))

    # for idx, value in enumerate(label_num):
    #     lt_label_num.append(tail_num + (max_num-tail_num)*imb_num_list[sorted_indices[idx]])
    lt_label_num = get_pareto_list(max_num,tail_num,alpha=alpha)
    lt_label_num = np.round(lt_label_num)


    # 使用numpy的argsort函数获取排序的索引
    sorted_indices = np.argsort(label_num)[::-1]

    # 根据排序后的索引对原始列表进行替换
    j = 0
    label_s = [0,0,0,0,0,0]
    for i in sorted_indices:
        label_s[i] = lt_label_num[j]
        j += 1

    # from matplotlib import pyplot as plt
    # plt.plot(np.arange(6),label_s)
    # plt.title(f"alpha={alpha}, beta={imbalance_factor}")
    # plt.show()
    new_signals_tr_np = []
    new_labels_tr_np = []
    sort_sig = []
    sort_label = []
    for idx, num in enumerate(label_s):
        sort_sig.append(signals[labels==idx])
        sort_label.append(labels[labels==idx])
        new_signals_tr_np.append(sort_sig[idx][:int(num),:])
        new_labels_tr_np.append(sort_label[idx][:int(num)])

    new_signals = np.vstack(new_signals_tr_np)
    new_labels = np.hstack(new_labels_tr_np)
    return new_signals,new_labels


def get_lt_data_rest(signals, labels, imbalance_factor=10, alpha=5):
    label_num = [0,0,0,0,0,0]
    imb_num_list = np.random.pareto(alpha, len(label_num))
    imb_num_list = sorted(imb_num_list)
    imb_num_list = normalize(imb_num_list)
    for label in labels:
        label_num[label] += 1
    label_num = np.array(label_num)


    max_num = max(label_num)
    min_num = min(label_num)
    tail_num = max_num/imbalance_factor
    # lt_label_num = np.round(max_num - ((max_num - tail_num)/max(sorted_indices)) * (max(sorted_indices)-np.array(sorted_indices)))

    # for idx, value in enumerate(label_num):
    #     lt_label_num.append(tail_num + (max_num-tail_num)*imb_num_list[sorted_indices[idx]])
    lt_label_num = get_pareto_list(max_num,tail_num,alpha=alpha)
    lt_label_num = np.round(lt_label_num)


    # 使用numpy的argsort函数获取排序的索引
    sorted_indices = np.argsort(label_num)[::-1]

    # 根据排序后的索引对原始列表进行替换
    j = 0
    label_s = [0,0,0,0,0,0]
    for i in sorted_indices:
        label_s[i] = lt_label_num[j]
        j += 1

    # from matplotlib import pyplot as plt
    # plt.plot(np.arange(6),label_s)
    # plt.title(f"alpha={alpha}, beta={imbalance_factor}")
    # plt.show()
    new_signals_tr_np = []
    new_labels_tr_np = []
    sort_sig = []
    sort_label = []
    for idx, num in enumerate(label_s):
        sort_sig.append(signals[labels==idx])
        sort_label.append(labels[labels==idx])
        new_signals_tr_np.append(sort_sig[idx][int(num):,:])
        new_labels_tr_np.append(sort_label[idx][int(num):])

    new_signals = np.vstack(new_signals_tr_np)
    new_labels = np.hstack(new_labels_tr_np)
    return new_signals,new_labels


def get_mean_data(signals, labels):
    label_num = [0,0,0,0,0,0]
    for label in labels:
        label_num[label] += 1
    label_num = np.array(label_num)
    max_num = max(label_num)
    min_num = min(label_num)

    mean_label_num = min_num*np.ones_like([0,0,0,0,0,0])

    new_signals_tr_np = []
    new_labels_tr_np = []
    sort_sig = []
    sort_label = []
    for idx, num in enumerate(mean_label_num):
        sort_sig.append(signals[labels==idx])
        sort_label.append(labels[labels==idx])
        new_signals_tr_np.append(sort_sig[idx][:int(num),:])
        new_labels_tr_np.append(sort_label[idx][:int(num)])

    new_signals = np.vstack(new_signals_tr_np)
    new_labels = np.hstack(new_labels_tr_np)
    return new_signals,new_labels


def create_base_dataset(file_name, train_frac, dim):
    frame = pd.read_table(file_name)
    train_fraction = train_frac
    signals_tr = []
    labels_tr = []
    signals_tt = []
    labels_tt = []
    addition_sample_num = 0
    for idx in range(len(frame)):
        mat_name = os.path.join('data/raw_data', frame['file_name'][idx])
        raw_data = scio.loadmat(mat_name)
        for key, value in raw_data.items():
            if key[5:7] == 'DE':
                signal = value
                sample_num = signal.shape[0] // dim
                train_num = int(sample_num * train_fraction)
                test_num = sample_num - train_num
                signal = signal[0:dim * sample_num]
                signals = np.array(np.split(signal, sample_num))

                signals_tr.append(signals[0:train_num, :])

                signals_tt.append(signals[train_num:sample_num, :])
                labels_tr.append(frame['label'][idx] * np.ones(train_num + addition_sample_num))
                labels_tt.append(frame['label'][idx] * np.ones(test_num))

    signals_tr_np = np.concatenate(signals_tr).squeeze()
    labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
    signals_tt_np = np.concatenate(signals_tt).squeeze()
    labels_tt_np = np.concatenate(np.array(labels_tt)).astype('uint8')
    return signals_tr_np, labels_tr_np, signals_tt_np, labels_tt_np


def create_cwru_dataset(train, ssv = False, ssv_size=100, excep_num=100, normal_num=500, train_frac=0.8):
    frame = pd.read_table('data/annotations.txt')
    dim = 1024
    train_fraction = train_frac
    # print(frame)
    signals_tr = []
    labels_tr = []
    signals_tt = []
    labels_tt = []
    count = 0
    addition_sample_num = 0
    for idx in range(len(frame)):
        mat_name = os.path.join('data/raw_data', frame['file_name'][idx])
        raw_data = scio.loadmat(mat_name)
        for key, value in raw_data.items():
            if key[5:7] == 'DE':
                signal = value
                sample_num = signal.shape[0] // dim
                train_num = int(sample_num * train_fraction)
                test_num = sample_num - train_num
                signal = signal[0:dim * sample_num]
                signals = np.array(np.split(signal, sample_num))

                signals_tr.append(signals[0:train_num, :])

                signals_tt.append(signals[train_num:sample_num, :])
                labels_tr.append(frame['label'][idx] * np.ones(train_num + addition_sample_num))
                labels_tt.append(frame['label'][idx] * np.ones(test_num))

    signals_tr_np = np.concatenate(signals_tr).squeeze()
    labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
    signals_tt_np = np.concatenate(signals_tt).squeeze()
    labels_tt_np = np.concatenate(np.array(labels_tt)).astype('uint8')

    signals_tr_np, signals_tt_np = normalize(signals_tr_np, signals_tt_np)

    signals_tr_np, labels_tr_np, ssv_set = train_set_split_ssv(signals_tr_np, labels_tr_np, ssv_size)

    '''注释的两行是构造长尾分布数据集用的'''
    signals_tr_np, labels_tr_np = get_imb_data(signals_tr_np, labels_tr_np, excep_num=excep_num, normal_num=normal_num)

    signals_tt_np,labels_tt_np = get_mean_data(signals_tt_np,labels_tt_np)
    # signals_tt_np,labels_tt_np = get_lt_data(signals_tt_np,labels_tt_np,imbalance_factor=100)
    print(signals_tr_np.shape, labels_tr_np.shape, signals_tt_np.shape, labels_tt_np.shape)
    if train:
        # return CWRUdata(signals_tr_np,labels_tr_np), CWRUdata(ssv_set,create_ssv_y(ssv_set))
        return CWRUdata(signals_tr_np, labels_tr_np)
        # ssv_set_fake, fake_y = create_ssv_y_fake_sample(ssv_set, ssv_set.shape[-1]/3)
        # return CWRUdata(signals_tr_np, labels_tr_np), CWRUdata(ssv_set_fake, fake_y)
    elif ssv:
        '''
            create_ssv_y得到标准化后的先验知识标签,
            create_dcae_label得到dcae提取的24维特征
            create_ae_label得到线性ae的特征            
         '''
        # return CWRUdata(ssv_set, create_ae_label(
        #     CWRUdata(ssv_set, np.zeros((ssv_set.shape[0], 1)))))
        dcae_label = create_dcae_label(
            CWRUdata(ssv_set, np.zeros((ssv_set.shape[0], 1))))
        mean = np.mean(dcae_label, axis=0)
        std_dev = np.std(dcae_label, axis=0)
        mean = mean.reshape((1, -1))
        std_dev = std_dev.reshape((1, -1))
        # 标准化矩阵
        dcae_label = (dcae_label - mean) / std_dev
        knowledge_label = create_knowledge_label(ssv_set)
        proxy_label = np.hstack((dcae_label, knowledge_label))
        return CWRUdata(ssv_set, proxy_label)
    else:
        return CWRUdata(signals_tt_np,labels_tt_np)
