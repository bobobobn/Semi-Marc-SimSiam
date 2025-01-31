import torch
import torch.nn as nn
import torch.optim as optim
import gym


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc_out(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# 初始化环境、策略网络和优化器
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
gamma = 0.99
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    episode_states = []
    episode_actions = []
    episode_rewards = []

    while True:
        # 状态转换为张量
        state_tensor = torch.from_numpy(state[0]).float().unsqueeze(0)

        # 获取动作概率分布并采样动作
        probs = policy_net(state_tensor)
        action = torch.multinomial(probs.squeeze(0), 1).item()

        # 执行动作并记录结果
        next_state, reward, done, _, _ = env.step(action)
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        if done:
            break
        state = next_state

    # 计算折扣回报
    discounted_rewards = []
    R = 0
    for r in reversed(episode_rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    # 归一化回报
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8)

    # 计算损失并更新策略网络
    optimizer.zero_grad()
    for state, action, G in zip(episode_states, episode_actions, discounted_rewards):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy_net(state_tensor)
        log_prob = torch.log(probs.squeeze(0)[action])  # 对应动作的 log 概率
        loss = -log_prob * G  # 损失是 log_prob 加权的回报
        loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}: Total Reward = {sum(episode_rewards)}")

env.close()
