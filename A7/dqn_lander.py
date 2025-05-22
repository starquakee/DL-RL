import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 尝试解决OMP库重复初始化警告
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import time

# 超参数
BUFFER_SIZE = int(1e5)  # 经验回放池大小
BATCH_SIZE = 64  # 小批量大小
GAMMA = 0.99  # 折扣因子
LR = 5e-4  # 学习率
UPDATE_EVERY = 4  # Q网络更新频率（每多少步）
TARGET_UPDATE_EVERY = 100  # 目标网络更新频率（每多少步）

# 设置设备 (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class QNetwork(nn.Module):
    """策略模型 (Q值网络)."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """初始化参数并构建模型。
        Params
        ======
            state_size (int): 状态维度
            action_size (int): 动作维度
            seed (int): 随机种子
            fc1_units (int): 第一个隐藏层的节点数
            fc2_units (int): 第二个隐藏层的节点数
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """构建一个将状态映射到动作价值的网络。"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """固定大小的缓冲区，用于存储经验元组。"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """初始化ReplayBuffer对象。"""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(
            seed)  # 注意: random.seed()返回None，这里应为 torch.manual_seed(seed) 或 np.random.seed(seed) 如果需要影响buffer的采样随机性

    def add(self, state, action, reward, next_state, done):
        """向内存中添加新的经验。"""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """从内存中随机采样一批经验。"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """返回内部内存的当前大小。"""
        return len(self.memory)


class DQNAgent():
    """与环境交互并从环境中学习的智能体。"""

    def __init__(self, state_size, action_size, seed):
        """初始化Agent对象。"""
        self.state_size = state_size
        self.action_size = action_size
        # self.seed = random.seed(seed) # random.seed() 返回 None, 通常在顶层设置或传递给torch/numpy
        if seed is not None:  # 确保seed被正确使用
            random.seed(seed)
            # np.random.seed(seed) # 如果numpy的随机性也需要被固定
            # torch.manual_seed(seed) # torch的随机种子在QNetwork中已设置

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step_update = 0
        self.t_step_target_update = 0

    def step(self, state, action, reward, next_state, done):
        """保存经验到回放记忆，并按需进行学习和目标网络更新。"""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step_update = (self.t_step_update + 1) % UPDATE_EVERY
        if self.t_step_update == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

        self.t_step_target_update = (self.t_step_target_update + 1) % TARGET_UPDATE_EVERY
        if self.t_step_target_update == 0:
            self.update_target_network(self.qnetwork_local, self.qnetwork_target)

    def act(self, state, eps=0.):
        """根据当前策略为给定状态返回动作。"""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 重命名以避免覆盖外部state
        self.qnetwork_local.eval()  # 设置网络为评估模式
        with torch.no_grad():
            action_values = self.qnetwork_local(state_t)
        self.qnetwork_local.train()  # 将网络设置回训练模式

        if random.random() > eps:  # epsilon-greedy 动作选择
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """使用给定的一批经验元组更新价值参数。"""
        states, actions, rewards, next_states, dones = experiences

        # 从目标模型获取下一状态的最大预测Q值
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # 计算当前状态的Q目标值: Y = r + gamma * Q_target(s', argmax_a' Q_local(s',a')) (这是DDQN的思想，标准DQN是 Y = r + gamma * max_a' Q_target(s',a'))
        # 此处实现的是标准DQN的目标：Y = r + gamma * max_a' Q_target(s',a')
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 从本地模型获取预期的Q值
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)  # 计算损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, local_model, target_model):
        """硬更新目标网络参数: θ_target = θ_local"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


def dqn_train(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """深度Q学习训练函数。"""
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    solved_at_episode = None

    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        terminated = False
        truncated = False
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.3f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.3f}')

        if solved_at_episode is None and np.mean(scores_window) >= 200.0:  # 检查是否解决，但不提前退出
            solved_at_episode = i_episode - 100 if i_episode >= 100 else i_episode
            print(
                f'\nEnvironment first solved around episode {solved_at_episode:d}! Current Avg Score: {np.mean(scores_window):.2f}')
            # 可在此处保存一个“已解决”状态的检查点模型
            # try:
            #     torch.save(agent.qnetwork_local.state_dict(), f'dqn_checkpoint_solved_ep{i_episode}.pth')
            #     print(f"Checkpoint saved at episode {i_episode}")
            # except Exception as e:
            #     print(f"Error saving checkpoint model: {e}")
    return scores


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"State size: {state_size}, Action size: {action_size}")

    dqn_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)  # seed=0 用于可复现性

    print("Starting DQN Training...")
    training_start_time = time.time()
    # 增加训练回合数以便更好地学习
    dqn_scores = dqn_train(env, dqn_agent, n_episodes=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
    training_time_dqn = time.time() - training_start_time
    print(f"DQN Training Finished. Time: {training_time_dqn:.2f} seconds.")

    # 保存最终训练的模型权重
    try:
        torch.save(dqn_agent.qnetwork_local.state_dict(), 'dqn_lander_weights.pth')
        print("DQN model weights saved to dqn_lander_weights.pth")
    except Exception as e:
        print(f"Error saving DQN model weights: {e}")

    # 绘制训练曲线并保存滚动平均数据
    try:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(np.arange(1, len(dqn_scores) + 1), dqn_scores, label='DQN Episode Scores')
        rolling_avg_dqn_for_plot = []
        if len(dqn_scores) >= 100:
            rolling_avg_dqn_for_plot = [np.mean(dqn_scores[max(0, i - 99):i + 1]) for i in range(len(dqn_scores))]
            plt.plot(np.arange(1, len(rolling_avg_dqn_for_plot) + 1), rolling_avg_dqn_for_plot,
                     label='DQN Rolling Average (100 episodes)', color='purple', alpha=0.7)
            # 保存滚动平均奖励数据，用于后续所有算法的对比图
            np.save('dqn_rolling_avg_scores.npy', np.array(rolling_avg_dqn_for_plot))
            print("SUCCESS: DQN rolling average scores saved to dqn_rolling_avg_scores.npy")

        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.title('DQN Training Progress on LunarLander-v2')
        plt.legend()
        plt.grid(True)
        plt.savefig('dqn_training_scores.png')
        print("SUCCESS: DQN training scores plot saved to dqn_training_scores.png")
        plt.close(fig)  # 显式关闭图像
    except Exception as e:
        print(f"ERROR during plotting or saving files: {e}")

    # 最终策略评估
    test_episodes = 100
    total_rewards_dqn_test = []
    landed_successfully_dqn = 0
    fuel_consumption_dqn_test = []

    print("\nTesting DQN Policy...")
    for i_test_episode in range(test_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_fuel_estimate = 0.0
        terminated = False
        truncated = False
        for t_step in range(1000):  # Max steps per episode
            action = dqn_agent.act(state, eps=0.0)  # 测试时使用贪婪策略
            # 估算燃料消耗
            if action == 2:
                episode_fuel_estimate += 0.3
            elif action == 1 or action == 3:
                episode_fuel_estimate += 0.03
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards_dqn_test.append(episode_reward)
        fuel_consumption_dqn_test.append(episode_fuel_estimate)
        if episode_reward > 200:
            landed_successfully_dqn += 1

    avg_reward_dqn_test = np.mean(total_rewards_dqn_test) if total_rewards_dqn_test else -float('inf')
    success_rate_dqn = (landed_successfully_dqn / test_episodes) * 100 if test_episodes > 0 else 0
    avg_fuel_dqn_test = np.mean(fuel_consumption_dqn_test) if fuel_consumption_dqn_test else 0

    print(f"\n--- DQN Final Policy Evaluation ---")
    print(f"Average Test Reward: {avg_reward_dqn_test:.2f}")
    print(f"Success Rate: {success_rate_dqn:.2f}%")
    print(f"Average Fuel Consumption (estimate): {avg_fuel_dqn_test:.2f} (points penalty based)")
    print(f"Total Training Time for DQN: {training_time_dqn:.2f} seconds")

    env.close()