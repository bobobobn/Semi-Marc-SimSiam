import torch
import random
import numpy as np

class ToTensor1D:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


class RandomCrop(torch.nn.Module):
    def __init__(self, crop_size_set, times=1):
        """
        随机裁剪音频信号的变换
        :param crop_size: 裁剪的大小
        """
        super().__init__()
        self.crop_size_set = crop_size_set
        self.times = times

    def generate_random_intervals(self, a, b, L, N, intervals=[]):
        """
        随机生成不重叠的区间
        :param a: 区间范围下界
        :param b: 区间范围上界
        :param L: 每个区间的长度
        :param N: 需要生成的区间数量
        :return: 不重叠的区间列表
        """
        if N * L > b - a:
            raise ValueError("区间范围不足以容纳所有不重叠的区间。")

        used_points = set()

        while len(intervals) < N:
            # 随机生成起点
            start = random.randint(a, b - L)
            # 检查是否与已有区间重叠
            if any(start < end and start + L > begin for begin, end in intervals):
                continue
            # 添加新的区间
            intervals.append((start, start + L))
            # 标记区间内的点为已使用
            used_points.update(range(start, start + L))

        # 返回排序后的区间（可选）
        return sorted(intervals)

    def forward(self, signal):
        """
        随机裁剪输入的音频信号
        :param signal: 输入音频波形，形状 [channels, time]
        :return: 随机裁剪后的音频信号
        """
        intervals = self.generate_random_intervals(0, len(signal)-1, self.crop_size_set[random.randint(0, len(self.crop_size_set)-1)], self.times)
        for i in range(self.times):
            start, end = intervals[i][0], intervals[i][1]
            # signal[start:end] = (signal[start] + signal[end]) * 0.5
            signal[start:end] = 0
        return signal




import torch
import torch.nn as nn


class RandomScaled(nn.Module):
    def __init__(self, scale_range=(0.8, 1.2)):
        """
        随机缩放音频信号的变换
        :param scale_range: 缩放因子的范围 (min_scale, max_scale)
        """
        super().__init__()
        self.scale_range = scale_range

    def forward(self, signal):
        """
        对输入信号进行随机缩放
        :param signal: 输入音频波形，形状 [time]
        :return: 缩放后的信号
        """

        # 使用 numpy 生成一个随机缩放因子
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        # 对信号进行缩放
        scaled_signal = signal * scale_factor
        return scaled_signal


class TimeShift(torch.nn.Module):
    def __init__(self, shift_limit):
        """
        随机平移音频信号的变换
        :param shift_limit: 平移的限制范围（最大平移的样本数）
        """
        super().__init__()
        self.shift_limit = shift_limit

    def forward(self, signal):
        """
        随机平移音频信号
        :param signal: 输入音频波形，形状 [channels, time]
        :return: 平移后的音频信号
        """
        shift = random.randint(-self.shift_limit, self.shift_limit)
        return np.roll(signal, shift)

class RandomAbs(nn.Module):
    def __init__(self, p=0.5):
        """
        随机对信号取绝对值的变换。
        :param p: 取绝对值的概率，0 <= p <= 1，默认值为 0.5。
        """
        super().__init__()
        self.p = p

    def forward(self, signal):
        """
        对输入信号按概率取绝对值。
        :param signal: 输入信号，形状为 [time] 或 [batch, time]。
        :return: 经过随机取绝对值处理后的信号。
        """

        if random.random() < self.p:
            return np.abs(signal)
        return signal

class RandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        """
        随机对信号取绝对值的变换。
        :param p: 取绝对值的概率，0 <= p <= 1，默认值为 0.5。
        """
        super().__init__()
        self.p = p

    def forward(self, signal):
        """
        对输入信号按概率取绝对值。
        :param signal: 输入信号，形状为 [time] 或 [batch, time]。
        :return: 经过随机取绝对值处理后的信号。
        """

        if random.random() < self.p:
            return -signal
        return signal


class TimeShift(torch.nn.Module):
    def __init__(self, shift_limit):
        """
        随机平移音频信号的变换
        :param shift_limit: 平移的限制范围（最大平移的样本数）
        """
        super().__init__()
        self.shift_limit = shift_limit

    def forward(self, signal):
        """
        随机平移音频信号
        :param signal: 输入音频波形，形状 [channels, time]
        :return: 平移后的音频信号
        """
        shift = random.randint(-self.shift_limit, self.shift_limit)
        return np.roll(signal, shift)

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.05):
        """
        添加高斯噪声的变换
        :param mean: 高斯噪声的均值 (默认 0.0)
        :param std: 高斯噪声的标准差 (默认 0.05)
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, waveform):
        """
        向波形添加高斯噪声
        :param waveform: 输入音频波形，形状 [channels, time]
        :return: 添加噪声后的波形
        """
        noise = np.random.randn(len(waveform)) * self.std + self.mean
        return waveform + noise

class AddGaussianNoiseSNR(torch.nn.Module):
    def __init__(self, snr=2):
        """
        添加高斯噪声的变换
        :param mean: 高斯噪声的均值 (默认 0.0)
        :param std: 高斯噪声的标准差 (默认 0.05)
        """
        super().__init__()
        self.snr_db = snr

    def forward(self, signal):
        """
        向波形添加高斯噪声
        :param waveform: 输入音频波形，形状 [channels, time]
        :return: 添加噪声后的波形
        """
        signal_power = np.mean(signal ** 2)

        # Compute noise power based on SNR
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

        # Add noise to the signal
        noisy_signal = signal + noise
        return noisy_signal




class WeightedMovingAverage(torch.nn.Module):
    def __init__(self, window_size, weights=None):
        """
        Initializes the Weighted Moving Average filter.

        :param window_size: Size of the moving window.
        :param weights: A list or tensor of weights to apply. If None, equal weights are used.
        """
        super().__init__()
        self.window_size = window_size

        if weights is None:
            # If no weights are provided, use uniform weights (equal weight to all)
            self.weights = torch.ones(window_size) / window_size
        else:
            self.weights = torch.tensor(weights)
            # Normalize weights so that their sum equals 1 (to maintain scale)
            self.weights = self.weights / self.weights.sum()

    def forward(self, signal):
        """
        Applies the weighted moving average filter to the input signal.

        :param signal: Input signal (tensor) with shape [channels, time]
        :return: The filtered signal
        """
        # Ensure the signal is a 2D tensor (channels, time)
        if signal.ndimension() != 2:
            raise ValueError("Input signal must be a 2D tensor (channels, time)")

        # Prepare the filtered signal tensor
        filtered_signal = torch.zeros_like(signal)

        # Apply weighted moving average filter
        for t in range(self.window_size - 1, signal.size(1)):
            # Get the window of the signal
            window = signal[:, t - self.window_size + 1:t + 1]
            # Apply weights and compute the weighted sum
            weighted_sum = torch.sum(window * self.weights.view(1, -1), dim=1)
            filtered_signal[:, t] = weighted_sum

        return filtered_signal

class GaussianWeightedMovingAverage(torch.nn.Module):
    def __init__(self, window_size, sigma):
        """
        初始化高斯加权移动平均滤波器
        :param window_size: 滤波器窗口大小
        :param sigma: 高斯分布的标准差
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        # 生成高斯权重
        self.weights = self.generate_gaussian_weights(window_size, sigma)

    def generate_gaussian_weights(self, window_size, sigma):
        """
        生成高斯权重
        :param window_size: 滤波器窗口大小
        :param sigma: 高斯分布的标准差
        :return: 归一化后的高斯权重数组
        """
        t = np.linspace(-window_size // 2, window_size // 2, window_size)
        weight = np.exp(-t**2 / (2 * sigma**2))  # 高斯权重

        # 归一化权重，使其总和为1
        weight /= np.sum(weight)
        return weight

    def forward(self, signal):
        """
        应用高斯加权移动平均滤波器
        :param signal: 输入信号，形状为 [time]
        :return: 滤波后的信号
        """
        # 确保输入信号是一维数组
        if signal.ndim != 1:
            raise ValueError("Input signal must be a 1D array (time)")

        # 使用 np.convolve 进行卷积操作，'same' 模式确保输出与输入形状相同
        filtered_signal = np.convolve(signal, self.weights, mode='same')
        return filtered_signal

class RandomChooseDA(torch.nn.Module):
    def __init__(self, da_set):
        super().__init__()
        self.da_set = da_set

    def forward(self, signal):
        return self.da_set[random.randint(0, len(self.da_set)-1)](signal)


class RandomChunkShuffle(torch.nn.Module):
    def __init__(self, num_chunks=10):
        super().__init__()
        self.num_chunks = num_chunks

    def forward(self, signal):
        signal_length = len(signal)
        chunk_size = signal_length // self.num_chunks  # 每块的大小



        # 分块
        chunks = np.array_split(signal, self.num_chunks)

        # 打乱块顺序
        np.random.shuffle(chunks)

        # 重组信号
        shuffled_signal = np.concatenate(chunks)

        return shuffled_signal

class RandomReverse(torch.nn.Module):
    def __init__(self, p=0.5):
        """
        随机反转音频信号
        :param p: 反转的概率
        """
        super().__init__()
        self.p = p

    def forward(self, signal):
        if random.random() < self.p:
            return signal[::-1].copy()
        return signal



class RandomNormalize(torch.nn.Module):
    def __init__(self, norm_range=(0.5, 1.5)):
        """
        随机归一化信号
        :param norm_range: 归一化范围 (min, max)
        """
        super().__init__()
        self.norm_range = norm_range

    def forward(self, signal):
        if random.random() < 0.5:
            norm_factor = np.random.uniform(self.norm_range[0], self.norm_range[1])
            return signal / np.max(np.abs(signal)) * norm_factor
        return signal


class PhasePerturbation(torch.nn.Module):
    def __init__(self, max_perturb=0.1):
        """
        随机扰动相位
        :param max_perturb: 最大扰动幅度 (弧度)
        """
        super().__init__()
        self.max_perturb = max_perturb

    def forward(self, signal):
        fft_signal = np.fft.fft(signal)
        phase = np.angle(fft_signal)
        magnitude = np.abs(fft_signal)

        # 添加随机相位扰动
        perturbed_phase = phase + np.random.uniform(-self.max_perturb, self.max_perturb, size=phase.shape)
        perturbed_fft = magnitude * np.exp(1j * perturbed_phase)
        return np.fft.ifft(perturbed_fft).real


from scipy.interpolate import interp1d

class RandomFrequencyWarp(torch.nn.Module):
    def __init__(self, warp_factor_range=(0.9, 1.1)):
        """
        随机频率扭曲
        :param warp_factor_range: 扭曲因子范围
        """
        super().__init__()
        self.warp_factor_range = warp_factor_range

    def forward(self, signal):
        warp_factor = np.random.uniform(self.warp_factor_range[0], self.warp_factor_range[1])
        original_indices = np.arange(len(signal))
        # 确保 warped_indices 和 signal 的长度一致
        warped_indices = np.linspace(0, len(signal) - 1, len(signal))
        warped_signal = interp1d(warped_indices * warp_factor, signal, kind='linear', fill_value="extrapolate")
        return warped_signal(original_indices)


class RandomlyAct(torch.nn.Module):
    def __init__(self, func, p=0.5):
        """
        随机反转音频信号
        :param p: 反转的概率
        """
        super().__init__()
        self.p = p
        self.func = func

    def forward(self, signal):
        if random.random() < self.p:
            return self.func(signal)
        return signal