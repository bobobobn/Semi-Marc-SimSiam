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
class encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(encoder, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.fc6 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc7 = nn.Sequential(
            nn.Linear(64, Output_size),)

    def forward(self, x):
        global inputflag
        if x.shape[2] == 512:
            inputflag = 0
            out = x.view(x.size(0), -1)
            out = self.fc1(out)
        else:
            inputflag = 1
            out = x.view(x.size(0), -1)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        return out


class decoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(decoder, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(Output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.fc5 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc6 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc7 = nn.Sequential(
            nn.Linear(1024, 512),)


    def forward(self, z):
        out = self.fc1(z)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        if inputflag == 0:
            out = self.fc6(out)
            out = self.fc7(out)
        else:
            out = self.fc6(out)

        return out


class classifier(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(classifier, self).__init__()
        self.fc6 = nn.Sequential(nn.ReLU(), nn.Linear(Output_size, out_channel))

    def forward(self, z):
        label = self.fc6(z)
        return label

def create_ae_label(ssv_set):
    ssv_loader = DataLoader(ssv_set, batch_size=opt.batch_size, shuffle=True)

    learningRate = 0.001
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
    for epoch in range(200):
        t0 = time.time()
        print('Starting epoch %d / %d' % (epoch + 1, opt.epochs))
        optimizer.step()
        # scheduler.step()
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
