import numpy as np
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import gym
from lunar_lander_utils import discretize_state  # 假设这是您的工具文件
import pickle

# 全局创建环境实例，供后续函数使用
env = gym.make('LunarLander-v2')


def monte_carlo_control(env, num_episodes, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.999, epsilon_min=0.01):
    """
    蒙特卡洛控制算法 (首次访问型) 用于 LunarLander-v2。

    Args:
        env: OpenAI Gym 环境。
        num_episodes (int): 训练的总回合数。
        gamma (float): 折扣因子。
        epsilon_start (float): 初始探索率。
        epsilon_decay (float): epsilon 每回合的衰减率。
        epsilon_min (float): epsilon 的最小值。

    Returns:
        Q (defaultdict): 学习到的Q值表。
        policy (defaultdict): 从Q值表导出的贪婪策略。
        episode_rewards (list): 每个训练回合的总奖励列表。
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = defaultdict(lambda: np.random.choice(env.action_space.n))
    episode_rewards = []
    epsilon = epsilon_start

    for i_episode in range(1, num_episodes + 1):
        episode_history = []  # 存储 (状态, 动作, 奖励)
        observation, info = env.reset()
        discretized_state = discretize_state(observation)
        terminated = False
        truncated = False
        current_episode_reward = 0

        # 生成一个完整的 episode
        while not (terminated or truncated):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[discretized_state])

            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_history.append((discretized_state, action, reward))
            current_episode_reward += reward
            discretized_state = discretize_state(next_observation)

        episode_rewards.append(current_episode_reward)

        # 更新 Q 值 (首次访问型蒙特卡洛)
        G = 0
        visited_state_actions = set()
        for t in range(len(episode_history) - 1, -1, -1):
            s_t_discrete, a_t, r_t = episode_history[t]
            G = gamma * G + r_t
            if (s_t_discrete, a_t) not in visited_state_actions:
                returns_sum[s_t_discrete][a_t] += G
                returns_count[s_t_discrete][a_t] += 1
                Q[s_t_discrete][a_t] = returns_sum[s_t_discrete][a_t] / returns_count[s_t_discrete][a_t]
                policy[s_t_discrete] = np.argmax(Q[s_t_discrete])  # 更新策略
                visited_state_actions.add((s_t_discrete, a_t))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if i_episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode {i_episode}/{num_episodes} - Avg Reward (last 100): {avg_reward:.2f} - Epsilon: {epsilon:.3f}")

    return Q, policy, episode_rewards


if __name__ == '__main__':
    mc_num_episodes = 20000

    print("Starting Monte Carlo Training...")
    start_time = time.time()
    Q_mc, policy_mc, mc_rewards = monte_carlo_control(env, num_episodes=mc_num_episodes)
    training_time_mc = time.time() - start_time
    print(f"Monte Carlo Training Finished. Time: {training_time_mc:.2f} seconds.")

    # 保存训练好的Q表
    try:
        with open('q_table_mc.pkl', 'wb') as f:
            pickle.dump(dict(Q_mc), f)  # 将 defaultdict 转换为 dict 保存
        print("Monte Carlo Q-table saved to q_table_mc.pkl")
    except Exception as e:
        print(f"Error saving Q-table for Monte Carlo: {e}")

    # 绘制训练曲线并保存滚动平均数据
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(mc_rewards) + 1), mc_rewards, label='Monte Carlo Episode Rewards')

    rolling_avg_mc_for_plot = []
    if len(mc_rewards) >= 100:
        rolling_avg_mc_for_plot = [np.mean(mc_rewards[max(0, i - 99):i + 1]) for i in range(len(mc_rewards))]
        plt.plot(np.arange(1, len(rolling_avg_mc_for_plot) + 1), rolling_avg_mc_for_plot,
                 label='MC Rolling Average (100 episodes)', color='blue', alpha=0.7)

        # 保存滚动平均奖励数据，用于后续所有算法的对比图
        np.save('monte_carlo_rolling_avg_rewards.npy', np.array(rolling_avg_mc_for_plot))
        print("Monte Carlo rolling average rewards saved to monte_carlo_rolling_avg_rewards.npy")

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Monte Carlo Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('monte_carlo_training_rewards.png')
    # plt.show()

    # 最终策略评估
    test_episodes = 100
    total_rewards_mc_test = []
    landed_successfully_mc = 0
    fuel_consumption_mc_test = []

    print("\nTesting Monte Carlo Policy...")
    for i_test_episode in range(test_episodes):
        observation, info = env.reset()
        discretized_s = discretize_state(observation)
        terminated = False
        truncated = False
        episode_reward = 0
        episode_fuel_estimate = 0.0

        for t_step in range(1000):  # LunarLander环境每局通常最多1000步
            action = policy_mc.get(discretized_s, env.action_space.sample())  # 使用get处理未见过的状态

            # 估算燃料消耗
            if action == 2:  # 主引擎
                episode_fuel_estimate += 0.3
            elif action == 1 or action == 3:  # 侧引擎
                episode_fuel_estimate += 0.03

            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            discretized_s = discretize_state(next_observation)

            if terminated or truncated:
                break

        total_rewards_mc_test.append(episode_reward)
        fuel_consumption_mc_test.append(episode_fuel_estimate)
        if episode_reward > 200:
            landed_successfully_mc += 1

    avg_reward_mc_test = np.mean(total_rewards_mc_test) if total_rewards_mc_test else -float('inf')
    success_rate_mc = (landed_successfully_mc / test_episodes) * 100 if test_episodes > 0 else 0
    avg_fuel_mc_test = np.mean(fuel_consumption_mc_test) if fuel_consumption_mc_test else 0

    print(f"\n--- Monte Carlo Final Policy Evaluation ---")
    print(f"Average Test Reward: {avg_reward_mc_test:.2f}")
    print(f"Success Rate: {success_rate_mc:.2f}%")
    print(f"Average Fuel Consumption (estimate): {avg_fuel_mc_test:.2f} (points penalty based)")
    print(f"Total Training Time for Monte Carlo: {training_time_mc:.2f} seconds")

    env.close()