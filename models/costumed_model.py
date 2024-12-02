# -*- coding: utf-8 -*-
'''
20180401
nn architecture for CWRU datasets of 101classification
BY rlk
'''


from torch import nn


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