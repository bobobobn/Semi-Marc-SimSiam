# -*- coding: utf-8 -*-
'''
20180401
nn architecture for CWRU datasets of 101classification
BY rlk
'''


from torch import nn
import torch

class Flatten(nn.Module):
    def forward(self, x):
        N, C, L = x.size()  # read in N, C, L
        z = x.view(N, -1)
#        print(C, L)
        return z  # "flatten" the C * L values into a single vector per image


class CWRUcnn(nn.Module):
    def __init__(self, kernel_num1=27, kernel_num2=27, kernel_size=55, pad=0, ms1=16, ms2=16, class_num=6):
        super(CWRUcnn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(1, kernel_num1, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, padding = pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            Flatten()
        )
        self.linear = nn.Linear(108, class_num)
        #ms1=16,ms2=16
#            nn.Linear(27*14, 101)) #ms1=16,ms2=9
#            nn.Linear(27*25, 101)) #ms1=9,ms2=9
#            nn.Linear(27*75, 101))  #ms1=9,ms2=3

    def forward(self, x):
        x = self.layers(x)
        return self.linear(x)

class CNN(nn.Module):
    def __init__(self, kernel_num1=32, kernel_num2=64, kernel_size=4, pad=0, ms1=16, ms2=16, class_num=6, feature_num=24,fine_tune = False):
        super(CNN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(1, kernel_num1, kernel_size, padding = 'same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding = 'same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, padding = 'same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, padding='same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(4096, feature_num)
        )
        #ms1=16,ms2=16
#            nn.Linear(27*14, 101)) #ms1=16,ms2=9
#            nn.Linear(27*25, 101)) #ms1=9,ms2=9
#            nn.Linear(27*75, 101))  #ms1=9,ms2=3
        self.fine_tune = fine_tune
    def forward(self, x):
        x = self.layers(x)
        return self.linear(x)

class CNN_Alfa(nn.Module):
    def __init__(self, kernel_num1=32, kernel_num2=64, kernel_size=4, pad=0, ms1=16, ms2=16, class_num=6, feature_num=24,fine_tune = False):
        super(CNN_Alfa, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(1, kernel_num1, kernel_size, padding = 'same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding = 'same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, padding = 'same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, padding='same', stride=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(384, feature_num)
        )
        #ms1=16,ms2=16
#            nn.Linear(27*14, 101)) #ms1=16,ms2=9
#            nn.Linear(27*25, 101)) #ms1=9,ms2=9
#            nn.Linear(27*75, 101))  #ms1=9,ms2=3
        self.fine_tune = fine_tune
    def forward(self, x):
        x = self.layers(x)
        return self.linear(x)


class CNN_Fine(nn.Module):
    def __init__(self, kernel_num1=27, kernel_num2=27, kernel_size=55, pad=0, ms1=16, ms2=16, class_num=6, feature_num=24,fine_tune = False):
        super(CNN_Fine, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(108, feature_num)
        )
        self.soft = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(feature_num, class_num)
        )
        #ms1=16,ms2=16
#            nn.Linear(27*14, 101)) #ms1=16,ms2=9
#            nn.Linear(27*25, 101)) #ms1=9,ms2=9
#            nn.Linear(27*75, 101))  #ms1=9,ms2=3
        self.fine_tune = fine_tune
    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x)
        return self.soft(x)

    def forward_without_fc(self, x):
        x = self.layers(x)
        x = self.linear(x)
        return self.soft(x)


class CNNEncoder(nn.Module):
    def __init__(self, num_classes, input_channels=1, hidden_size=256, num_blocks=10, zero_init_residual=False, kernel_num1=27, kernel_num2=27, kernel_size=55, pad=0, ms1=16, ms2=16,):
        """
        A CNN encoder with stacked Conv1d blocks.

        :param input_channels: Number of input channels for the first convolutional layer
        :param hidden_size: Number of kernels (output channels) in each convolutional layer
        :param num_blocks: Number of stacked convolutional blocks
        """
        super(CNNEncoder, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.layers = nn.Sequential(
            nn.Conv1d(1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num1, kernel_num1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num1),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel_num1, kernel_num2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel_num2, kernel_num2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel_num2),
            nn.ReLU(),
            Flatten()
        )
        if zero_init_residual:
            nn.init.zeros_(self.layers[-3].weight)
        self.fc = nn.Linear(108, num_classes)
        # ms1=16,ms2=16
        #            nn.Linear(27*14, 101)) #ms1=16,ms2=9
        #            nn.Linear(27*25, 101)) #ms1=9,ms2=9
        #            nn.Linear(27*75, 101))  #ms1=9,ms2=3

    def forward(self, x):
        x = self.layers(x)
        return self.fc(x)


class StackedCNNEncoderWithPooling(nn.Module):
    def __init__(self, num_classes, zero_init_residual=False, input_channels=1, hidden_size=256, num_blocks=10, pooling="max"):
        """
        Encoder with stacked CNN blocks and pooling layers.

        :param input_channels: Number of input channels for the first convolutional layer
        :param hidden_size: Number of kernels (output channels) in each convolutional layer
        :param num_blocks: Number of stacked convolutional blocks
        :param pooling: Pooling method: "avg" for average pooling, "max" for max pooling
        """
        super(StackedCNNEncoderWithPooling, self).__init__()

        layers = []
        for i in range(num_blocks):
            layers.append(
                nn.Conv1d(
                    in_channels=input_channels if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            # Add pooling layer
            if pooling == "avg":
                layers.append(nn.AvgPool1d(kernel_size=2, stride=2))  # Average pooling
            elif pooling == "max":
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # Max pooling
        layers.append(Flatten())
        # if zero_init_residual:
        #     nn.init.constant_(layers[-3].weight, 0)
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.fc.in_features = 256
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass of the encoder with pooling.

        :param x: Input tensor of shape (batch_size, input_channels, sequence_length)
        :return: Encoded features
        """
        x = self.encoder(x)
        x = self.fc(x)
        return x


    def forward_without_fc(self, x):
        """
        Forward pass of the encoder with pooling.

        :param x: Input tensor of shape (batch_size, input_channels, sequence_length)
        :return: Encoded features
        """
        x = self.encoder(x)
        return x


class StackedCNNEncoderWithPoolingNoFC(nn.Module):
    def __init__(self, num_classes, zero_init_residual=False, input_channels=1, hidden_size=256, num_blocks=10, pooling="avg"):
        """
        Encoder with stacked CNN blocks and pooling layers.

        :param input_channels: Number of input channels for the first convolutional layer
        :param hidden_size: Number of kernels (output channels) in each convolutional layer
        :param num_blocks: Number of stacked convolutional blocks
        :param pooling: Pooling method: "avg" for average pooling, "max" for max pooling
        """
        super(StackedCNNEncoderWithPoolingNoFC, self).__init__()

        layers = []
        for i in range(num_blocks):
            layers.append(
                nn.Conv1d(
                    in_channels=input_channels if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    padding=1,
                    stride=1
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))

            # Add pooling layer
            if pooling == "avg":
                layers.append(nn.AvgPool1d(kernel_size=16, stride=1))  # Average pooling
            elif pooling == "max":
                layers.append(nn.MaxPool1d(kernel_size=16, stride=1))  # Max pooling
        layers.append(Flatten())
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        """
        Forward pass of the encoder with pooling.

        :param x: Input tensor of shape (batch_size, input_channels, sequence_length)
        :return: Encoded features
        """
        x = self.encoder(x)
        return x

from torch.nn import functional as F

class ClassBalancedLoss(torch.nn.Module):
    def __init__(self, class_counts, beta=0.99):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.class_counts = class_counts

    def forward(self, logits, labels):
        """
        logits: 模型的输出 (batch_size, num_classes)
        labels: 真实标签 (batch_size)
        class_counts: 每个类别的样本数量 (num_classes)
        """
        effective_num = 1.0 - torch.pow(self.beta, self.class_counts)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum()  # 归一化权重
        weights = weights.cuda()
        # 转换为与 logits 对应的 batch 权重
        label_weights = weights[labels]

        loss = F.cross_entropy(logits, labels, reduction='none')
        loss = label_weights * loss  # 加权交叉熵
        return loss.mean()
