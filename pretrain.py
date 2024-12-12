# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:31:08 2018

@author: rlk
"""

from config import Config
from data import create_dataset
from models import create_model
from torch.utils.data import DataLoader
from utils import check_accuracy
import torch
from tensorboardX import SummaryWriter
import copy
import time
import pandas as pd
from models import MLP
from models import Resnet1d
from models import Alexnet1d
from models import BiLSTM1d
from models import LeNet1d
from torchsummary import summary
from models import costumed_model
import numpy as np
opt = Config()
from data import data_preprocess
from data import alfaDataProcess
from data import ssv_data

def save_checkpoint(acc, epoch, state):
    filename = f'./checkpoints/epoch{epoch}_acc{acc}ckpt.pth.tar'
    torch.save(state, filename)


def remove_outliers(array, threshold=1):
    # 计算每列的均值和标准差
    mean = np.mean(array, axis=0)
    std_dev = np.std(array, axis=0)

    # 计算 Z 分数
    z_scores = np.abs((array - mean) / std_dev)

    # 根据阈值删除极端值
    filtered_array = array[(z_scores < threshold)]

    return filtered_array


def make_agent_dataset(dataset_tr, dataset_val, class_num=6):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=class_num, random_state=42)
    X = np.vstack((dataset_tr.X,dataset_val.X))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(dataset_tr.X)
    l={}
    for v in y_kmeans:
        if v not in l.keys():
            l[v] = 1
        else:
            l[v] += 1
    dataset_tr.y = kmeans.predict(dataset_tr.X)
    dataset_val.y = kmeans.predict(dataset_val.X)


original_data = ssv_data.KnowledgeSSVData(ssv_size=200, normal_size=500, excep_size=100)
ssv_set = original_data.get_ssv()
class_num = ssv_set.y.shape[-1]
# ssv_set = data_preprocess.create_cwru_dataset(train=False, ssv=True, ssv_size=100, excep_num=100, normal_num=500, train_frac=0.8)
ssv_loader = DataLoader(ssv_set, batch_size=opt.batch_size, shuffle=True)
# print('#training num = %d' % len(tr_dataset))

# print('#val num = %d' % len(val_dataset))

# model = costumed_model.CNN_Alfa(feature_num=24, class_num=9)
model = costumed_model.CWRUcnn(class_num=class_num)
# model = Resnet1d.resnet18()
# model = MLP.MLP()
# model = Alexnet1d.alexnet()
# model = BiLSTM1d.BiLSTM()
# model = LeNet1d.LeNet()
writer = SummaryWriter(comment=str(opt.model_param['kernel_num1'])+'_'+
                       str(opt.model_param['kernel_num2']))

total_steps = 0

#选择优化器
learningRate = 0.001

optimizer = torch.optim.Adam(params=model.parameters(),
                           lr=learningRate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters,
                                            opt.lr_decay)  # regulation rate decay

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#损失函数BCELoss - 不包含sigmoid
loss_fn = torch.nn.MSELoss()

###==============training=================###

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA is available') 
device = torch.device(opt.device if use_cuda else "cpu")
model = model.to(device)
#summary(model, (1,2048))
# save best_model wrt. val_acc
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
# one epoch
data_loss = []
val_acc_list = []
for epoch in range(opt.epochs):
    t0 = time.time()
    print('Starting epoch %d / %d' % (epoch + 1, opt.epochs))
    optimizer.step()
    scheduler.step()
    # set train model or val model for BN and Dropout layers
    model.train()
    # one batch
    for t, (x, y) in enumerate(ssv_loader):
        # add one dim to fit the requirements of conv1d layer
        x.resize_(x.size()[0], 1, x.size()[1]) 
        x, y = x.float(), y.float()
        x, y = x.to(device), y.to(device)
        # loss and predictions
        scores = model(x)
        loss = loss_fn(scores, y)
        data_loss.append(loss.to("cpu").detach().numpy())
        writer.add_scalar('loss', loss.item())
        # print and save loss per 'print_every' times
        if (t + 1) % opt.print_every == 0:
            print('t = %d, loss = %.4f' % (t + 1, loss.item()))
        # parameters update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    save_checkpoint(loss,epoch,{
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
        })
    t1 = time.time()
    print(t1-t0)

import matplotlib.pyplot as plt
# data_loss = remove_outliers(np.array(data_loss))
plt.plot(range(len(data_loss)),data_loss)
plt.xlabel(u'steps')
plt.ylabel(u'loss')
plt.show()
# plt.plot(range(len(val_acc_list)),val_acc_list)
# plt.xlabel(u'steps')
# plt.ylabel(u'val acc')
# plt.show()
print('kernel num1: {}'.format(opt.model_param['kernel_num1']))
print('kernel num2: {}'.format(opt.model_param['kernel_num2']))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
val_acc, confuse_matrix = check_accuracy(model, val_loader, device, error_analysis=True)
# write the confuse_matrix to Excel
data_pd = pd.DataFrame(confuse_matrix)
writer = pd.ExcelWriter('results\\confuse_matrix_rate.xlsx')
data_pd.to_excel(writer)
writer.save()
writer.close()
# save model in results dir
model_save_path = 'results\\' + time.strftime('%Y%m%d%H%M_') + str(int(100*best_acc)) + '.pth'
torch.save(model.state_dict(), model_save_path)
print('best model is saved in: ', model_save_path)

# 此段代码是一段模型训练代码，主要实现了以下功能：
#
# 1. 加载配置，创建训练集和验证集，创建模型，初始化优化器，定义损失函数，指定设备；
#
# 2. 使用Adam优化器和学习率衰减来训练模型，打印每个batch的损失，同时记录到Tensorboard中；
#
# 3. 每个epoch结束后，计算验证集和训练集的准确率，并记录到Tensorboard中，比较验证集准确率，若最新准确率高于之前最高准确率，则保存最新模型；
#
# 4. 加载最佳模型，计算验证集准确率，并将混淆矩阵存储到Excel中，最后保存模型。



