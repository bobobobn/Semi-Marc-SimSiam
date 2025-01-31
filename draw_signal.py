from data.ssv_data import *
from matplotlib import pyplot as plt
from simsiam.DA.data_augmentations import *


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（适用于 Windows 系统，如需其他系统请调整字体）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.pyplot as plt

# 帕累托分布参数
class_num = 10      # 类的数量（区间数量）
from scipy.optimize import fsolve

# 给定最多类与最少类的比值 R，以及类编号 n
R = 100  # 最大类与最小类的比值
n = class_num   # 最少类对应的区间编号

# 定义方程组，根据 R 和 n 求解 alpha
def equation(alpha, R, n):
    p1 = 1 - 2**(-alpha)
    pn = n**(-alpha) - (n+1)**(-alpha)
    return p1 / pn - R

# 初始猜测 alpha 值
alpha_guess = 1.0

# 使用 fsolve 求解 alpha
alpha = fsolve(equation, alpha_guess, args=(R, n))[0]

# 输出结果
print(f"通过最大类与最小类比值 R={R} 和类编号 n={n}，计算得到的 alpha 值为：{alpha:.4f}")

# 定义区间边界
bins = np.arange(1, class_num + 2)  # 区间 [0, 1], [1, 2], ..., [class_num-1, class_num]

# 定义帕累托分布的累积分布函数 F(x)
def pareto_cdf(x, alpha):
    return 1 - x**(-alpha)

# 计算每个区间的积分
cdf_values = pareto_cdf(bins, alpha)  # 从 1 开始计算 CDF 值 # 在 0 位置插入 0（F(0) = 0）
interval_probs = np.diff(cdf_values)      # 每个区间的积分

# 确保占比总和为 1
interval_probs /= interval_probs.sum()

# 打印每类的积分占比
print("每类的积分占比（从小到大）：")
for i, prob in enumerate(interval_probs):
    print(f"区间 [{bins[i]}, {bins[i+1]}): {prob:.4f}")

# 绘制每类的占比图
plt.figure(figsize=(10, 6))
plt.bar(range(len(interval_probs)), interval_probs, color='skyblue', edgecolor='black', alpha=0.7)
plt.xticks(range(len(interval_probs)), [f"[{bins[i]}, {bins[i+1]})" for i in range(len(interval_probs))], rotation=45)
plt.title('帕累托分布离散化区间的占比', fontsize=16)
plt.xlabel('区间', fontsize=12)
plt.ylabel('占比', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 定义求解 α 的函数
def solve_alpha(beta, n=10):
    """
    通过最大类与最小类样本数比值 beta 计算帕累托分布的 α。
    """
    def equation(alpha):
        numerator = 1 - 2**(-alpha)
        denominator = n**(-alpha) - (n + 1)**(-alpha)
        return beta - numerator / denominator

    alpha_initial_guess = 10.0
    alpha_solution = fsolve(equation, alpha_initial_guess)[0]
    return alpha_solution

# 设置参数范围
x = np.linspace(1, 10, 500)  # x 的范围
beta_values = [1,2,3,5]    # 不同的 β 值
n_max = 10

# 绘制不同 β 的帕累托分布概率密度函数
plt.figure(figsize=(10, 6))
for beta in beta_values:
    alpha = solve_alpha(beta, n=n_max)  # 计算对应的 α 值

    pdf = alpha * x ** (-alpha - 1)     # 帕累托概率密度函数
    plt.plot(x, pdf, label=f"$\\beta={beta}, \\alpha={alpha:.2f}$")

# 设置图形
plt.title("帕累托分布概率密度函数 (不同 $\\beta$)", fontsize=14)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("概率密度", fontsize=12)
plt.legend(fontsize=12)
plt.show()





#
# # 初始化数据对象
# nonLabeled = NonLabelSSVData()
#
# # 获取训练数据
# train_data = nonLabeled.get_train()
#
# # 提取特征数据
# X = train_data.X
# augmentation = [
#     AddGaussianNoiseSNR(snr=6),
#     TimeShift(512),
#     RandomChunkShuffle(30),
#     RandomCrop([5], 100),
#     RandomScaled((0.5, 1.5)),
# ]
# sec_augmentation = [
#     AddGaussianNoiseSNR(snr=6),
#     RandomNormalize(),
#     PhasePerturbation(0.2),
#     RandomChunkShuffle(30),
#     RandomCrop([5], 100),
#     RandomScaled((0.5, 1.5)),
#     RandomAbs(),
#     RandomVerticalFlip(),
#     RandomReverse(),
# ]
# # 绘制 X[0] 时域信号
# plt.plot(PhasePerturbation(0.4)(X[0]), color='#000000')
# # 设置纵坐标范围
#
# plt.title("Time-Domain Signal of X[0]")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.show()
