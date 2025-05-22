import pickle
import time
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import gym
from lunar_lander_utils import discretize_state  # 假设这是您的工具文件

# 全局创建环境实例，供后续函数使用
env = gym.make('LunarLander-v2')


def q_learning_algorithm(env, num_episodes, alpha=0.1, gamma=0.99,
                         epsilon_start=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
    """
    Q-learning 控制算法用于 LunarLander-v2。

    Args:
        env: OpenAI Gym 环境。
        num_episodes (int): 训练的总回合数。
        alpha (float): 学习率。
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
    policy = defaultdict(lambda: np.random.choice(env.action_space.n))  # 初始策略
    episode_rewards = []
    epsilon = epsilon_start

    for i_episode in range(1, num_episodes + 1):
        observation, info = env.reset()
        discretized_state = discretize_state(observation)
        current_episode_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # 探索
            else:
                action = np.argmax(Q[discretized_state])  # 利用

            next_observation, reward, terminated, truncated, info = env.step(action)
            discretized_next_state = discretize_state(next_observation)
            current_episode_reward += reward

            # Q-learning 更新规则: Q(S,A) <- Q(S,A) + alpha * [R + gamma*max_a' Q(S',a') - Q(S,A)]
            best_next_action_q_value = np.max(Q[discretized_next_state]) if not (terminated or truncated) else 0
            td_target = reward + gamma * best_next_action_q_value
            td_error = td_target - Q[discretized_state][action]
            Q[discretized_state][action] += alpha * td_error

            discretized_state = discretized_next_state

        episode_rewards.append(current_episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if i_episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode {i_episode}/{num_episodes} - Avg Reward (last 100): {avg_reward:.2f} - Epsilon: {epsilon:.3f}")

    for s_q_key in Q:  # 为所有遇到过的状态生成最终的贪婪策略
        policy[s_q_key] = np.argmax(Q[s_q_key])

    return Q, policy, episode_rewards


if __name__ == '__main__':
    q_learning_num_episodes = 20000
    q_learning_alpha = 0.1

    print("Starting Q-learning Training...")
    start_time = time.time()
    Q_q_learning, policy_q_learning, q_learning_rewards = q_learning_algorithm(
        env,
        num_episodes=q_learning_num_episodes,
        alpha=q_learning_alpha
    )
    training_time_q_learning = time.time() - start_time
    print(f"Q-learning Training Finished. Time: {training_time_q_learning:.2f} seconds.")

    # 保存训练好的Q表
    try:
        with open('q_table_q_learning.pkl', 'wb') as f:
            pickle.dump(dict(Q_q_learning), f)  # 将 defaultdict 转换为 dict 保存
        print("Q-learning Q-table saved to q_table_q_learning.pkl")
    except Exception as e:
        print(f"Error saving Q-table for Q-learning: {e}")

    # 绘制训练曲线并保存滚动平均数据
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(q_learning_rewards) + 1), q_learning_rewards, label='Q-learning Episode Rewards')

    rolling_avg_q_learning_for_plot = []
    if len(q_learning_rewards) >= 100:
        rolling_avg_q_learning_for_plot = [np.mean(q_learning_rewards[max(0, i - 99):i + 1]) for i in
                                           range(len(q_learning_rewards))]
        plt.plot(np.arange(1, len(rolling_avg_q_learning_for_plot) + 1), rolling_avg_q_learning_for_plot,
                 label='Q-learning Rolling Average (100 episodes)', color='red', alpha=0.7)

        # 保存滚动平均奖励数据，用于后续所有算法的对比图
        np.save('q_learning_rolling_avg_rewards.npy', np.array(rolling_avg_q_learning_for_plot))
        print("Q-learning rolling average rewards saved to q_learning_rolling_avg_rewards.npy")

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-learning Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('q_learning_training_rewards.png')
    # plt.show()

    # 最终策略评估
    test_episodes = 100
    total_rewards_q_learning_test = []
    landed_successfully_q_learning = 0
    fuel_consumption_q_learning_test = []

    print("\nTesting Q-learning Policy...")
    for i_test_episode in range(test_episodes):
        observation, info = env.reset()
        discretized_s = discretize_state(observation)
        terminated = False
        truncated = False
        episode_reward = 0
        episode_fuel_estimate = 0.0

        for t_step in range(1000):  # LunarLander环境每局通常最多1000步
            action = policy_q_learning.get(discretized_s, env.action_space.sample())  # 使用get处理未见过的状态

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

        total_rewards_q_learning_test.append(episode_reward)
        fuel_consumption_q_learning_test.append(episode_fuel_estimate)
        if episode_reward > 200:
            landed_successfully_q_learning += 1

    avg_reward_q_learning_test = np.mean(total_rewards_q_learning_test) if total_rewards_q_learning_test else -float(
        'inf')
    success_rate_q_learning = (landed_successfully_q_learning / test_episodes) * 100 if test_episodes > 0 else 0
    avg_fuel_q_learning_test = np.mean(fuel_consumption_q_learning_test) if fuel_consumption_q_learning_test else 0

    print(f"\n--- Q-learning Final Policy Evaluation ---")
    print(f"Average Test Reward: {avg_reward_q_learning_test:.2f}")
    print(f"Success Rate: {success_rate_q_learning:.2f}%")
    print(f"Average Fuel Consumption (estimate): {avg_fuel_q_learning_test:.2f} (points penalty based)")
    print(f"Total Training Time for Q-learning: {training_time_q_learning:.2f} seconds")

    env.close()