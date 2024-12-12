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
import numpy as np
opt = Config()
import gModel
import dModel
from matplotlib import pyplot as plt
from tsne import plot_tsne
from imblearn.over_sampling import SMOTE
from data import data_preprocess, alfaDataProcess
from models import costumed_model
from data import ssv_data

# tr_dataset = alfaDataProcess.create_alfa_dataset(train=True, ssv_size=100, imbalance_factor=5)
# val_dataset = alfaDataProcess.create_alfa_dataset(train=False, ssv_size=100, imbalance_factor=5)

learingRate = 0.001
epochs = 50
import simsiam.DA.data_augmentations as data_augmentations


augmentation = [
    data_augmentations.TimeShift(256),
    data_augmentations.RandomCrop(100),
    data_augmentations.AddGaussianNoise(),
    # DA.data_augmentations.WeightedMovingAverage,
    data_augmentations.GaussianWeightedMovingAverage(5, 1),
]
import data.ssv_data as ssv_data
import torchvision.transforms as transforms

nonLabelCWRUData = ssv_data.NonLabelSSVData(ssv_size=100, normal_size=100, excep_size=100)
tr_dataset = nonLabelCWRUData.get_train()
val_dataset = nonLabelCWRUData.get_test()

pretrained = False
pretrained_model = r"C:\Users\bobobob\Desktop\1D-CNN-for-CWRU-master\checkpoints\checkpoint_0199.pth.tar"
# original_data = ssv_data.DataBase(ssv_size=200, normal_size=200, excep_size=200)

# tr_dataset = original_data.get_train()
# tr_dataset = data_preprocess.create_cwru_dataset(train=True, ssv_size=100, excep_num=500, normal_num=500, train_frac=0.8)
tr_loader = DataLoader(tr_dataset, batch_size=opt.batch_size, shuffle=True)
# val_dataset = original_data.get_test()
# val_dataset = data_preprocess.create_cwru_dataset(train=False,train_frac=0.8)
val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
print('#training num = %d' % len(tr_dataset))

# print('#val num = %d' % len(val_dataset))
# model = costumed_model.CNN_Alfa(feature_num=24, class_num=9)
# model = costumed_model.CNN_Fine(class_num=6, feature_num=32, fine_tune=True)
# model = MLP.MLP()
import models.Resnet1d as resnet
# model = resnet.resnet18(num_classes=6)
model = costumed_model.StackedCNNEncoderWithPooling(num_classes=10)
# model = Alexnet1d.alexnet()
# model = BiLSTM1d.BiLSTM()
# model = LeNet1d.LeNet()
writer = SummaryWriter(comment=str(opt.model_param['kernel_num1'])+'_'+
                       str(opt.model_param['kernel_num2']))

total_steps = 0
#选择优化器
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
#                            lr=learingRate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters,
#                                             opt.lr_decay)  # regulation rate decay

# Define the optimizer

from torch.optim.lr_scheduler import ExponentialLR
initial_lr = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

# Define the exponential decay scheduler
gamma = 0.95  # Decay factor
scheduler = ExponentialLR(optimizer, gamma=gamma)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#导入模型

#损失函数BCELoss - 不包含sigmoid
loss_fn = torch.nn.CrossEntropyLoss()

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

if pretrained:
    # checkpoint = torch.load(pretrained_model, map_location=torch.device(f'cuda:0'))
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #     # if 'linear' not in k and 'fc' not in k:
    #     new_state_dict[k] = v
    # model.load_state_dict(new_state_dict, strict=False)
    # print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')

    for name, param in model.named_parameters():
        if not name.startswith('fc'):
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
    model = model.to(device)


for epoch in range(epochs):
    t0 = time.time()
    print('Starting epoch %d / %d' % (epoch + 1, epochs))
    optimizer.step()
    scheduler.step()
    # set train model or val model for BN and Dropout layers
    model.train()
    # one batch
    for t, (x, y) in enumerate(tr_loader):
        # add one dim to fit the requirements of conv1d layer
        x.resize_(x.size()[0], 1, x.size()[1]) 
        x, y = x.float(), y.long()
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
    # save epoch loss and acc to train or val history
    train_acc, _= check_accuracy(model, tr_loader, device)
    val_acc, _= check_accuracy(model, val_loader, device)
    val_acc_list.append(val_acc)
    # writer acc and weight to tensorboard
    writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    # save the best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    t1 = time.time()

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
model = model.to(device)
X = torch.tensor(val_dataset.X).float()
X.resize_(X.size()[0], 1, X.size()[1])
X = X.to(device)
X = model.forward_without_fc(X).to('cpu').detach().numpy()
# 降维
X_tsne = tsne.fit_transform(X)
y = val_dataset.y
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
plt.colorbar(scatter, label='Classes')
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

plt.plot(range(len(data_loss)),data_loss)
plt.xlabel(u'steps')
plt.ylabel(u'loss')
plt.show()

plt.plot(range(len(val_acc_list)),val_acc_list)
plt.xlabel(u'steps')
plt.ylabel(u'val acc')
plt.show()
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



