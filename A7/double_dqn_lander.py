import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
        # 为当前网络层设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
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
        if seed is not None:  # 为buffer的采样过程设置随机种子
            random.seed(seed)

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
    """与环境交互并从环境中学习的智能体 (实现Double DQN)。"""

    def __init__(self, state_size, action_size, seed):
        """初始化Agent对象。"""
        self.state_size = state_size
        self.action_size = action_size
        if seed is not None:  # 为Agent内部的随机过程（如epsilon-greedy中的random.random()）设置种子
            random.seed(seed)

        # Q网络 (本地网络) 和目标网络
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.update_target_network(self.qnetwork_local, self.qnetwork_target)  # 确保初始权重一致

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
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()  # 设置网络为评估模式进行推理
        with torch.no_grad():
            action_values = self.qnetwork_local(state_t)
        self.qnetwork_local.train()  # 将网络设置回训练模式

        if random.random() > eps:  # epsilon-greedy 动作选择
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """使用给定的一批经验元组更新价值参数 (实现Double DQN)。"""
        states, actions, rewards, next_states, dones = experiences

        # --- Double DQN核心修改 ---
        # 1. 使用本地网络 qnetwork_local 选择在 next_states 中使得Q值最大的动作的索引
        local_next_actions_indices = self.qnetwork_local(next_states).detach().argmax(dim=1, keepdim=True)

        # 2. 使用目标网络 qnetwork_target 来评估这些被本地网络选出的动作的Q值
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(dim=1, index=local_next_actions_indices)
        # --- Double DQN核心修改结束 ---

        # 计算当前状态的Q目标值: Y = r + gamma * Q_target_next * (1 - done)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 从本地模型获取在实际采取的动作上的Q值 (预测值)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)  # 计算均方误差损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, local_model, target_model):
        """硬更新目标网络参数: θ_target = θ_local"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


def dqn_train(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
              algorithm_name="DQN"):
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
        eps = max(eps_end, eps_decay * eps)  # 更新epsilon值
        print(
            f'\r{algorithm_name} - Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.3f}',
            end="")
        if i_episode % 100 == 0:
            print(
                f'\r{algorithm_name} - Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.3f}')

        if solved_at_episode is None and np.mean(scores_window) >= 200.0:  # 检查是否解决环境，但不提前退出
            solved_at_episode = i_episode - 100 if i_episode >= 100 else i_episode
            print(
                f'\n{algorithm_name} - Environment first solved around episode {solved_at_episode:d}! Current Avg Score: {np.mean(scores_window):.2f}')
    return scores


if __name__ == '__main__':
    ALGORITHM_NAME = "DoubleDQN"
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"State size: {state_size}, Action size: {action_size}")

    # 使用DQNAgent类，但其内部learn方法已是Double DQN实现
    ddqn_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)

    print(f"Starting {ALGORITHM_NAME} Training...")
    training_start_time = time.time()
    # 根据需要调整训练回合数
    ddqn_scores = dqn_train(env, ddqn_agent, n_episodes=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                            algorithm_name=ALGORITHM_NAME)
    training_time_ddqn = time.time() - training_start_time
    print(f"{ALGORITHM_NAME} Training Finished. Time: {training_time_ddqn:.2f} seconds.")

    # 保存最终训练的模型权重
    weights_filename = f'{ALGORITHM_NAME.lower()}_lander_weights.pth'
    try:
        torch.save(ddqn_agent.qnetwork_local.state_dict(), weights_filename)
        print(f"{ALGORITHM_NAME} model weights saved to {weights_filename}")
    except Exception as e:
        print(f"Error saving {ALGORITHM_NAME} model weights: {e}")

    # 绘制训练曲线并保存滚动平均数据
    plot_filename = f'{ALGORITHM_NAME.lower()}_training_scores.png'
    npy_filename = f'{ALGORITHM_NAME.lower()}_rolling_avg_scores.npy'
    try:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(np.arange(1, len(ddqn_scores) + 1), ddqn_scores, label=f'{ALGORITHM_NAME} Episode Scores')
        rolling_avg_ddqn_for_plot = []
        if len(ddqn_scores) >= 100:
            rolling_avg_ddqn_for_plot = [np.mean(ddqn_scores[max(0, i - 99):i + 1]) for i in range(len(ddqn_scores))]
            plt.plot(np.arange(1, len(rolling_avg_ddqn_for_plot) + 1), rolling_avg_ddqn_for_plot,
                     label=f'{ALGORITHM_NAME} Rolling Average (100 episodes)', color='teal', alpha=0.7)  # 使用不同颜色

            np.save(npy_filename, np.array(rolling_avg_ddqn_for_plot))
            print(f"SUCCESS: {ALGORITHM_NAME} rolling average scores saved to {npy_filename}")

        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.title(f'{ALGORITHM_NAME} Training Progress on LunarLander-v2')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_filename)
        print(f"SUCCESS: {ALGORITHM_NAME} training scores plot saved to {plot_filename}")
        plt.close(fig)
    except Exception as e:
        print(f"ERROR during plotting or saving files for {ALGORITHM_NAME}: {e}")

    # 最终策略评估
    test_episodes = 100
    total_rewards_ddqn_test = []
    landed_successfully_ddqn = 0
    fuel_consumption_ddqn_test = []

    print(f"\nTesting {ALGORITHM_NAME} Policy...")
    for i_test_episode in range(test_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_fuel_estimate = 0.0
        terminated = False
        truncated = False
        for t_step in range(1000):
            action = ddqn_agent.act(state, eps=0.0)  # 测试时使用贪婪策略
            if action == 2:  # 估算燃料消耗
                episode_fuel_estimate += 0.3
            elif action == 1 or action == 3:
                episode_fuel_estimate += 0.03
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards_ddqn_test.append(episode_reward)
        fuel_consumption_ddqn_test.append(episode_fuel_estimate)
        if episode_reward > 200:
            landed_successfully_ddqn += 1

    avg_reward_ddqn_test = np.mean(total_rewards_ddqn_test) if total_rewards_ddqn_test else -float('inf')
    success_rate_ddqn = (landed_successfully_ddqn / test_episodes) * 100 if test_episodes > 0 else 0
    avg_fuel_ddqn_test = np.mean(fuel_consumption_ddqn_test) if fuel_consumption_ddqn_test else 0

    print(f"\n--- {ALGORITHM_NAME} Final Policy Evaluation ---")
    print(f"Average Test Reward: {avg_reward_ddqn_test:.2f}")
    print(f"Success Rate: {success_rate_ddqn:.2f}%")
    print(f"Average Fuel Consumption (estimate): {avg_fuel_ddqn_test:.2f} (points penalty based)")
    print(f"Total Training Time for {ALGORITHM_NAME}: {training_time_ddqn:.2f} seconds")

    env.close()