# -*- coding = utf-8 -*-
# @Time : 2023/5/24 20:26
# @Author : bobobobn
# @File : process_uestc_lab_data.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
import h5py
import pandas as pd
import numpy as np
import os
import scipy.io as scio


data_path_uestc = 'uestc_data/'
file_names = os.listdir(data_path_uestc)
frame = pd.read_table('annotations.txt')
dim = 2048
train_fraction = 0.8
# print(frame)
signals_tr = []
labels_tr = []
signals_tt = []
labels_tt = []
count = 0
# random_idx = [52459,2857,67930,74719,54299,60619,59451,31378,52438,13695,56484,2547,22154,3694,7771,65877,55586,25368,76018,2756]
addition_sample_num = 0
for file_name in file_names:
    if file_name[0] =='i':
        label = 1
    elif file_name[0] =='o':
        label = 2
    elif file_name[0] == 'n':
        label = 0
    data = np.loadtxt(data_path_uestc + file_name)
    signal = data[:, 1]
    sample_num = signal.shape[0] // dim
    train_num = int(sample_num * train_fraction)
    test_num = sample_num - train_num
    signal = signal[0:dim * sample_num]
    signals = np.array(np.split(signal, sample_num))
    signals_tr.append(signals[0:train_num, :])
    signals_tt.append(signals[train_num:sample_num, :])
    labels_tr.append(label * np.ones(train_num + addition_sample_num))
    labels_tt.append(label * np.ones(test_num))


signals_tr_np = np.concatenate(signals_tr).squeeze()
labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
signals_tt_np = np.concatenate(signals_tt).squeeze()
labels_tt_np = np.concatenate(np.array(labels_tt)).astype('uint8')
print(signals_tr_np.shape, labels_tr_np.shape, signals_tt_np.shape, labels_tt_np.shape)

f = h5py.File('DE.h5', 'w')
f.create_dataset('X_train', data=signals_tr_np)
f.create_dataset('y_train', data=labels_tr_np)
f.create_dataset('X_test', data=signals_tt_np)
f.create_dataset('y_test', data=labels_tt_np)
f.close()



