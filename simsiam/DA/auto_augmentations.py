import torch
import random
import numpy as np

import torch
import random
import numpy as np
import torch.nn as nn

def TakeDice(p):
    if random.random() < p:
        return True
    return False


class ToTensor1D:
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


class RandomCrop(torch.nn.Module):
    def __init__(self, crop_size, times=100, p=1.0):
        super().__init__()
        self.crop_size = int(crop_size)
        self.times = times
        self.p = p

    def generate_random_intervals(self, a, b, L, N, intervals=[]):
        if N * L > b - a:
            raise ValueError("区间范围不足以容纳所有不重叠的区间。")

        used_points = set()

        while len(intervals) < N:
            start = random.randint(a, b - L)
            if any(start < end and start + L > begin for begin, end in intervals):
                continue
            intervals.append((start, start + L))
            used_points.update(range(start, start + L))

        return sorted(intervals)

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        intervals = self.generate_random_intervals(0, len(signal)-1, self.crop_size, self.times)
        for i in range(self.times):
            start, end = intervals[i][0], intervals[i][1]
            signal[start:end] = 0
        return signal


class RandomScaled(nn.Module):
    def __init__(self, scale=0.2, p=1.0):
        super().__init__()
        self.scale_range = (1.0-scale, 1.0+scale)
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        scaled_signal = signal * scale_factor
        return scaled_signal


class TimeShift(torch.nn.Module):
    def __init__(self, shift_limit, p=1.0):
        super().__init__()
        self.shift_limit = int(shift_limit)
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        shift = random.randint(-self.shift_limit, self.shift_limit)
        return np.roll(signal, shift)


class RandomAbs(nn.Module):
    def __init__(self, scale=None, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        return np.abs(signal)


class RandomVerticalFlip(nn.Module):
    def __init__(self, scale=None, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        return -signal


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.05, p=1.0):
        super().__init__()
        self.mean = 0
        self.std = std
        self.p = p

    def forward(self, waveform):
        if not TakeDice(self.p):
            return waveform
        noise = np.random.randn(len(waveform)) * self.std + self.mean
        return waveform + noise


class AddGaussianNoiseSNR(torch.nn.Module):
    def __init__(self, snr=2, p=1.0):
        super().__init__()
        self.snr_db = snr
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        noisy_signal = signal + noise
        return noisy_signal


class RandomChunkShuffle(torch.nn.Module):
    def __init__(self, num_chunks=10, p=0.5):
        super().__init__()
        self.num_chunks = int(num_chunks)
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        signal_length = len(signal)
        chunk_size = signal_length // self.num_chunks
        chunks = np.array_split(signal, self.num_chunks)
        np.random.shuffle(chunks)
        shuffled_signal = np.concatenate(chunks)
        return shuffled_signal


class RandomReverse(torch.nn.Module):
    def __init__(self, scale=None, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        return signal[::-1].copy()


class RandomNormalize(torch.nn.Module):
    def __init__(self, scale=0.5, p=0.5):
        super().__init__()
        self.norm_range = (1.0-scale, 1.0+scale)
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        norm_factor = np.random.uniform(self.norm_range[0], self.norm_range[1])
        return signal / np.max(np.abs(signal)) * norm_factor


class PhasePerturbation(torch.nn.Module):
    def __init__(self, max_perturb=0.1, p=0.5):
        super().__init__()
        self.max_perturb = max_perturb
        self.p = p

    def forward(self, signal):
        if not TakeDice(self.p):
            return signal
        fft_signal = np.fft.fft(signal)
        phase = np.angle(fft_signal)
        magnitude = np.abs(fft_signal)
        perturbed_phase = phase + np.random.uniform(-self.max_perturb, self.max_perturb, size=phase.shape)
        perturbed_fft = magnitude * np.exp(1j * perturbed_phase)
        return np.fft.ifft(perturbed_fft).real


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.05, p = 1.0):
        """
        添加高斯噪声的变换
        :param mean: 高斯噪声的均值 (默认 0.0)
        :param std: 高斯噪声的标准差 (默认 0.05)
        """
        super().__init__()
        self.mean = 0
        self.std = std

    def forward(self, waveform):
        """
        向波形添加高斯噪声
        :param waveform: 输入音频波形，形状 [channels, time]
        :return: 添加噪声后的波形
        """
        if not TakeDice(self.p):
            return waveform
        noise = np.random.randn(len(waveform)) * self.std + self.mean
        return waveform + noise


class AddGaussianNoiseSNR(torch.nn.Module):
    def __init__(self, snr=2, p=1.0):
        """
        添加高斯噪声的变换
        :param mean: 高斯噪声的均值 (默认 0.0)
        :param std: 高斯噪声的标准差 (默认 0.05)
        """
        super().__init__()
        self.snr_db = snr
        self.p = p

    def forward(self, signal):
        """
        向波形添加高斯噪声
        :param waveform: 输入音频波形，形状 [channels, time]
        :return: 添加噪声后的波形
        """
        if not TakeDice(self.p):
            return signal
        signal_power = np.mean(signal ** 2)

        # Compute noise power based on SNR
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)

        # Add noise to the signal
        noisy_signal = signal + noise
        return noisy_signal



class SubPolicy():
    def __init__(self, policy, scales=None, need_p=True):
        self.policy = policy
        self.scales = scales
        self.needp = need_p


    def need_scale(self):
        return self.scales is not None


    def need_p(self):
        return self.needp

    def get_entity(self, scale=None, p=10):
        s = None
        if self.scales is not None:
            s = self.scales[0] + scale*(self.scales[1]-self.scales[0])*0.1
        return self.policy(s, p=p*0.1)