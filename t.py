import torch
import torch.nn.functional as F

# 假设有 3 个样本，每个样本有 4 个动作的概率
probs = torch.tensor([[0.1, 0.3, 0.4, 0.2],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.7, 0.2, 0.05, 0.05]])

# 创建离散概率分布
action_dist = torch.distributions.Categorical(probs)

# 随机采样动作
action = action_dist.sample()

print("概率分布:", probs)
print("采样的动作索引:", action)
