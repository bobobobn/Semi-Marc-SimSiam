import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, policies):
        super(Controller, self).__init__()
        self.policies = policies
        self.lstm_units = 100
        self.subpolicies = 5
        self.subpolicy_ops = 1
        self.op_probs = 11
        self.op_magnitudes = 11
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.lstm_units, num_layers=1, batch_first=True)
        self.dense_layers = nn.ModuleList()
        for i in range(self.subpolicy_ops):
            self.dense_layers.append(nn.Linear(self.lstm_units, self.op_probs))
            self.dense_layers.append(nn.Linear(self.lstm_units, self.op_magnitudes))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        outputs = []
        for i in range(self.subpolicy_ops):
            op_prob = self.dense_layers[i*3](lstm_out)
            op_magnitude = self.dense_layers[i*3 + 1](lstm_out)
            outputs.append(op_prob)
            outputs.append(op_magnitude)
        return outputs

    def fit(self, mem_softmaxes, mem_accuracies, optimizer, criterion):
        # 计算 mem_acc 的标准化范围
        min_acc = np.min(mem_accuracies)
        max_acc = np.max(mem_accuracies)
        if min_acc == max_acc:
            scales = [1.0] * len(mem_accuracies)  # 如果 min_acc == max_acc，所有权重均为 1
        else:
            scales = [(acc - min_acc) / (max_acc - min_acc) for acc in mem_accuracies]

        # 遍历历史输出和对应分数
        for softmaxes, scale in zip(mem_softmaxes, scales):
            optimizer.zero_grad()

            # 如果 softmaxes 是旧的网络输出，需重新构造为张量
            outputs = [torch.tensor(s, dtype=torch.float32, requires_grad=True) for s in softmaxes]

            # 使用损失函数计算每个输出的误差
            loss = 0
            for i, output in enumerate(outputs):
                target = torch.tensor(softmaxes[i], dtype=torch.float32)  # 目标值
                loss += criterion(output, target) * scale  # 按 acc 权重调整误差

            # 反向传播与优化
            loss.backward()
            optimizer.step()

    def predict(self, size):
        dummy_input = torch.zeros(1, size, 1)
        softmaxes = self(dummy_input)
        # Convert softmaxes into subpolicies
        subpolicies = []
        pscales = []
        for i in range(size):
            p = torch.argmax(softmaxes[0][0][i])
            scale = torch.argmax(softmaxes[1][0][i])
            subpolicies.append(self.policies[i%10].get_entity(scale=int(scale), p=int(p)))
            pscales.append((int(scale), int(p)))
        return softmaxes, subpolicies


class RandController(nn.Module):
    def __init__(self, policies):
        super(RandController, self).__init__()
        self.policies = policies
        self.lstm_units = 100
        self.subpolicies = 5
        self.subpolicy_ops = 1
        self.op_probs = 11
        self.op_magnitudes = 11
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.lstm_units, num_layers=1, batch_first=True)
        self.dense_layers = nn.ModuleList()
        for i in range(self.subpolicy_ops):
            self.dense_layers.append(nn.Linear(self.lstm_units, self.op_probs))
            self.dense_layers.append(nn.Linear(self.lstm_units, self.op_magnitudes))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        outputs = []
        for i in range(self.subpolicy_ops):
            op_prob = self.dense_layers[i*3](lstm_out)
            op_magnitude = self.dense_layers[i*3 + 1](lstm_out)
            probs = F.softmax(op_prob, dim=-1)  # 转换为概率分布
            outputs.append(probs)
            magnitudes = F.softmax(op_magnitude, dim=-1)  # 转换为概率分布
            outputs.append(magnitudes)
        return outputs

    def fit(self, size, actions, scores, optimizer):
        discounted_rewards = []
        R = 0
        gamma = 1.0
        for r in reversed(scores):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        # 归一化回报
        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-8)
        optimizer.zero_grad()
        for action, G in zip(actions, discounted_rewards):
            dummy_input = torch.zeros(size, 1)
            probs = self(dummy_input)
            log_prob =  torch.log(probs[0][torch.arange(probs[0].size(0)), action[0]]) + torch.log(probs[1][torch.arange(probs[1].size(0)), action[1]])
            loss = -log_prob.mean() * G  # 损失是 log_prob 加权的回报
            loss.backward()
        optimizer.step()
        return sum(scores)

    def predict(self, size):
        dummy_input = torch.zeros(size, 1)
        softmaxes = self(dummy_input)
        # Convert softmaxes into subpolicies
        subpolicies = []
        actions = []
        p = torch.multinomial(softmaxes[0], num_samples=1).squeeze(-1)
        scale = torch.multinomial(softmaxes[1], num_samples=1).squeeze(-1)
        for i in range(len(self.policies)):
            subpolicies.append(self.policies[i%len(self.policies)].get_entity(scale=int(scale[i]), p=int(p[i])))
        return softmaxes, subpolicies, [p, scale]