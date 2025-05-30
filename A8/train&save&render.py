import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 处理 OpenMP 库冲突问题
import gym as gym  # 使用 Gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque


# --- REINFORCE 算法类定义 ---
class REINFORCE:
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.99, device='cpu'):
        self.gamma = gamma
        self.device = device
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),  # 简单的两层网络
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        """根据当前策略选择一个动作"""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)  # log_prob 是一个 tensor
        return action.item(), log_prob  # 返回动作和该动作的对数概率

    def store_outcome(self, log_prob, reward):
        """存储一个时间步的奖励和对应动作的对数概率"""
        self.rewards.append(reward)
        self.saved_log_probs.append(log_prob)

    def update(self):
        """更新策略网络参数"""
        if not self.rewards:  # 如果 rewards 为空则不更新
            self.clear_memory()  # 清理以防万一
            return

        R = 0
        policy_loss = []
        returns = deque()  # 使用 deque 更高效地在前端插入

        # 从后往前计算每个时间步的折扣回报 G_t
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.appendleft(R)

        returns_tensor = torch.tensor(list(returns), dtype=torch.float32).to(self.device)

        # 标准化回报以减少方差 (可选，但通常有益)
        if len(returns_tensor) > 1:  # 只有一个元素时，std为0，会导致NaN
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
        elif len(returns_tensor) == 1:  # 只有一个元素，均值为自身，std为0
            returns_tensor = returns_tensor - returns_tensor.mean()  # 结果为0，但避免了NaN

        for log_prob, R_t in zip(self.saved_log_probs, returns_tensor):
            policy_loss.append(-log_prob * R_t)

        self.optimizer.zero_grad()
        if policy_loss:  # 确保 policy_loss 不为空
            policy_loss_tensor = torch.stack(policy_loss).sum()
            policy_loss_tensor.backward()
            self.optimizer.step()

        self.clear_memory()  # 清空当前 episode 的数据

    def clear_memory(self):
        """清除已存储的奖励和对数概率"""
        del self.rewards[:]
        del self.saved_log_probs[:]


# --- 超参数和设置 ---
ENV_NAME = "LunarLander-v2"
LEARNING_RATE = 0.01  # 根据你的实验结果，0.01对REINFORCE效果较好
GAMMA = 0.99
# 根据你的日志，REINFORCE (LR=0.01) 在1000-1200 episodes 左右表现不错
# 为了确保得到一个好的模型，可以训练稍长一些，比如1500 episodes
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1000
MODEL_SAVE_PATH = "reinforce_lunarlander_lr001.pth"  # 模型保存路径
SEED = 42  # 用于可复现性

# 设置设备 (GPU或CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 主训练逻辑 ---
def train_reinforce():
    print(f"Starting training for REINFORCE with LR={LEARNING_RATE} for {NUM_EPISODES} episodes...")

    # 设置随机种子
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)  # 为动作空间也设置种子

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, action_dim, learning_rate=LEARNING_RATE, gamma=GAMMA, device=device)

    episode_rewards = deque(maxlen=100)  # 用于记录最近100个episode的奖励以计算平均值

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)  # 为每个episode的环境重置设置不同的种子
        agent.clear_memory()
        current_episode_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_outcome(log_prob, reward)
            current_episode_reward += reward
            state = next_state

            if done or truncated:
                break

        agent.update()  # 在每个episode结束后更新网络
        episode_rewards.append(current_episode_reward)

        if episode % 50 == 0:  # 每50个episode打印一次进度
            avg_reward = np.mean(episode_rewards)
            print(f"Episode: {episode}/{NUM_EPISODES}, Average Reward (last 100): {avg_reward:.2f}")
            # 如果想提前停止，可以在这里加入判断条件，例如 avg_reward > 200
            if avg_reward > 230 and len(episode_rewards) >= 100:  # 例如，如果平均奖励很高了
                print(f"Achieved high average reward {avg_reward:.2f}. Stopping early.")
                break

    # 保存训练好的模型
    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    env.close()


# --- 加载模型并进行渲染的示例函数 ---
def load_and_render_reinforce(model_path, env_name, num_episodes=3):
    print(f"\nLoading model from {model_path} and rendering...")

    env = gym.make(env_name, render_mode="human")  # 设置 render_mode="human" 进行可视化
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 重新创建一个相同结构的模型实例
    # 注意：这里需要使用与训练时相同的learning_rate和gamma来初始化REINFORCE对象，
    # 但实际上对于推理，这些参数不重要，重要的是policy_net的结构和加载的权重。
    # device也应该与加载模型时使用的设备一致。
    agent = REINFORCE(state_dim, action_dim, learning_rate=LEARNING_RATE, gamma=GAMMA, device=device)

    # 加载模型权重
    # 需要确保模型在加载权重前与权重保存时的设备一致，或者使用 map_location
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()  # 设置为评估模式 (这对于有Dropout, BatchNorm层的网络很重要)

    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        print(f"\nRendering Episode {i + 1}")
        while not done and not truncated:
            env.render()
            with torch.no_grad():
                action, _ = agent.select_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        print(f"Episode {i + 1} finished. Reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    # train_reinforce()

    load_and_render_reinforce(MODEL_SAVE_PATH, ENV_NAME)