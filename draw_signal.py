from data.ssv_data import *
from matplotlib import pyplot as plt
from simsiam.DA.data_augmentations import *

import numpy as np
import os

# 设置中文字体（适用于 Windows 系统，如需其他系统请调整字体）
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 设置字体大小，适应论文格式
rcParams['font.size'] = 18  # 设置全局字体大小，调整此值即可

# 初始化数据对象
nonLabeled = NonLabelSSVData()

# 获取训练数据
train_data = nonLabeled.get_train()

# 提取特征数据
X = train_data.X
y = train_data.y

# 类标签
class_labels = ['NC', 'IF(0.007)', 'BF(0.007)', 'OF(0.007)', 'IF(0.014)',
                'BF(0.014)', 'OF(0.014)', 'IF(0.021)', 'BF(0.021)', 'OF(0.021)']

# 创建一个文件夹用来保存图像
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# 采样频率和采样周期
sampling_rate = 12000  # 12 kHz
sampling_period = 1 / sampling_rate  # 采样周期

# 遍历每个类标签
for label in range(10):
    # 获取当前类别的第一个样本索引
    index = np.where(y == label)[0][0]  # 获取第一个属于该类别的索引

    # 归一化信号到[-1, 1]范围
    signal = X[index]
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    signal_normalized = 2 * ((signal - signal_min) / (signal_max - signal_min)) - 1

    # 生成时间轴
    time_axis = np.arange(len(signal)) * sampling_period  # 计算时间轴，单位为秒

    # 绘制归一化后的信号
    plt.figure(figsize=(6, 6))
    plt.plot(time_axis, signal_normalized)
    plt.title(f'{class_labels[label]}', fontsize=25)  # 增大标题字体
    plt.xlabel('Time (s)', fontsize=25)  # 增大X轴标签字体
    plt.ylabel('Vibration (g)', fontsize=25)  # 增大Y轴标签字体

    plt.yticks([-1, 0, 1], ['-1', '0', '1'])  # 设定Y轴刻度

    # 保存图像
    plt.savefig(os.path.join(output_dir, f'class_{label}.png'))
    plt.close()
