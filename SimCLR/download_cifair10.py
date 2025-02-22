# -*- coding = utf-8 -*-
# @Time : 2024/12/22 18:54
# @Author : bobobobn
# @File : download_cifair10.py
# @Software: PyCharm
# 数据转换方式
from torchvision import transforms

# CIFAR10下载接口
from torchvision.datasets import CIFAR10

# 可视化
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import ssl
print(ssl.OPENSSL_VERSION)
my_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = CIFAR10(r"C:\Users\bobobob\Desktop\data", train=True, transform=my_trans, download=True)
test_dataset = CIFAR10(r"C:\Users\bobobob\Desktop\data", train=False, transform=my_trans, download=True)

