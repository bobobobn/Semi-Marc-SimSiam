from __future__ import print_function
import torch.nn as nn
import torch
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
from data import data_preprocess
from matplotlib import pyplot as plt
opt = Config()

Output_size = 24
inputflag = 0

# ----------------------------------inputsize == 1024
class Flatten(nn.Module):
    def forward(self, x):
        N, C, L = x.size()  # read in N, C, L
        z = x.view(N, -1)
#        print(C, L)
        return z  # "flatten" the C * L values into a single vector per image

class Unflatten(nn.Module):
    """Unflatten layer to reshape 2D tensor back to desired dimensions."""
    def __init__(self, channels, length):
        super(Unflatten, self).__init__()
        self.channels = channels
        self.length = length

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.length)

class encoder(nn.Module):
    def __init__(self, kernel_num1=32, kernel_num2=64, kernel_size=4, pad=0, ms1=16, ms2=16, class_num=6, feature_num=24,fine_tune = False):
        super(encoder, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(1, kernel_num1, kernel_size, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(192, feature_num)
        )
    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x)
        return x


# class decoder(nn.Module):
#     def __init__(self, kernel_num1=32, kernel_num2=64, kernel_size=4, pad=0, ms1=16, ms2=16, class_num=6, feature_num=24,fine_tune = False):
#         super(decoder, self).__init__()
#         pad = int((kernel_size - 1) / 2)
#         self.linear = nn.Linear(feature_num, 192)  # 和 Encoder 中的 Linear 对应
#
#         self.deconv_layers = nn.Sequential(
#             Unflatten(channels=64, length=3),  # 初始尺寸 (B, 64, 3)
#             nn.ConvTranspose1d(kernel_num2, kernel_num2, kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.ConvTranspose1d(kernel_num2, kernel_num1, kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.ConvTranspose1d(kernel_num1, kernel_num1, kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.ConvTranspose1d(kernel_num1, 1, kernel_size=4, stride=2)
#         )
#
#     def forward(self, x):
#         x = self.linear(x)  # 将特征向量映射到卷积层输入尺寸
#         return self.deconv_layers(x)
class decoder(nn.Module):
    def __init__(self, feature_num=24, kernel_num1=32, kernel_num2=64, kernel_size=4):
        super(decoder, self).__init__()

        # Linear层将特征向量映射回卷积输入形状
        self.linear = nn.Sequential(
            nn.Linear(feature_num, 192)
        )

        # 解码器的反卷积层 (ConvTranspose1d) 与 encoder 对应
        self.deconv_layers = nn.Sequential(
            Unflatten(64, 3),  # 初始形状 (batch_size, 64, 3)
            nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2),  # 逆向MaxPool1d
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2),  # 逆向MaxPool1d
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2),  # 逆向MaxPool1d
            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2, output_padding=1),  # 逆向MaxPool1d
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2)
        )

    def forward(self, x):
        x = self.linear(x)  # 线性层映射到卷积输入
        x = self.deconv_layers(x)  # 使用反卷积逐步恢复形状
        return x


def create_dcae_label(ssv_set):
    ssv_loader = DataLoader(ssv_set, batch_size=opt.batch_size, shuffle=True)

    learningRate = 0.0001
    epochs = 500
    encode = encoder()
    decode = decoder()
    params_a = list(encode.parameters())
    params_b = list(decode.parameters())

    # 合并参数列表
    all_params = params_a + params_b
    optimizer = torch.optim.Adam(params=all_params,
                                 lr=learningRate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters,
                                                opt.lr_decay)  # regulation rate decay
    loss_fn = torch.nn.MSELoss()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA is available')
    device = torch.device(opt.device if use_cuda else "cpu")
    encode = encode.to(device)
    decode = decode.to(device)

    # summary(model, (1,2048))
    # save best_model wrt. val_acc
    best_acc = 0.0
    # one epoch
    data_loss = []
    val_acc_list = []
    for epoch in range(epochs):
        t0 = time.time()
        print('Starting epoch %d / %d' % (epoch + 1, opt.epochs))
        optimizer.step()
        scheduler.step()
        # set train model or val model for BN and Dropout layers
        encode.train()
        decode.train()
        # one batch
        for t, (x, y) in enumerate(ssv_loader):
            # add one dim to fit the requirements of conv1d layer
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.float()
            x, y = x.to(device), y.to(device)
            # loss and predictions
            output = decode(encode(x))
            loss = loss_fn(output, x)
            data_loss.append(loss.to("cpu").detach().numpy())
            # writer.add_scalar('loss', loss.item())
            # print and save loss per 'print_every' times
            if (t + 1) % opt.print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.item()))
            # parameters update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t1 = time.time()
        print(t1 - t0)
    plt.plot(range(len(data_loss)), data_loss)
    plt.xlabel(u'steps')
    plt.ylabel(u'loss')
    # plt.show()
    x_set = ssv_set.X
    x_set = torch.tensor(x_set).to(device).float()
    x_set.resize_(x_set.size()[0], 1, x_set.size()[1])
    out = encode(x_set)
    return out.cpu().detach().numpy()
