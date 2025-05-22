import pickle
import time
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import gym
from lunar_lander_utils import discretize_state

env = gym.make('LunarLander-v2')


def sarsa_lambda_algorithm(env, num_episodes, alpha=0.1, gamma=0.99, lambda_trace=0.9,
                           epsilon_start=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                           trace_type='replacing', scheme_id='original'):
    """
    SARSA(lambda) control algorithm with eligibility traces for LunarLander-v2.
    ... (docstring 其他部分不变) ...
        trace_type (str): Type of eligibility trace ('replacing' or 'accumulating').
        scheme_id (str): Discretization scheme to use ('original', 'scheme1', 'scheme2').
    ...
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = defaultdict(lambda: np.random.choice(env.action_space.n))
    episode_rewards = []
    epsilon = epsilon_start

    for i_episode in range(1, num_episodes + 1):
        E = defaultdict(lambda: np.zeros(env.action_space.n))
        observation, info = env.reset()
        discretized_state = discretize_state(observation, scheme_id=scheme_id)

        current_episode_reward = 0
        terminated = False
        truncated = False

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[discretized_state])

        while not (terminated or truncated):
            next_observation, reward, terminated, truncated, info = env.step(action)
            # 使用传入的 scheme_id 进行离散化
            discretized_next_state = discretize_state(next_observation, scheme_id=scheme_id)
            current_episode_reward += reward

            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[discretized_next_state])

            td_target = reward + gamma * Q[discretized_next_state][next_action] * (not (terminated or truncated))
            td_error = td_target - Q[discretized_state][action]

            if trace_type == 'replacing':
                E[discretized_state][action] = 1.0
            elif trace_type == 'accumulating':
                E[discretized_state][action] += 1.0

            for s_e_key in list(E.keys()):
                for a_e_idx in range(env.action_space.n):
                    if E[s_e_key][a_e_idx] > 1e-6:
                        Q[s_e_key][a_e_idx] += alpha * td_error * E[s_e_key][a_e_idx]
                        E[s_e_key][a_e_idx] *= gamma * lambda_trace

            discretized_state = discretized_next_state
            action = next_action

        episode_rewards.append(current_episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if i_episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Scheme: {scheme_id} - Episode {i_episode}/{num_episodes} - Avg Reward (last 100): {avg_reward:.2f} - Epsilon: {epsilon:.3f}")

    for s_q_key in Q:
        policy[s_q_key] = np.argmax(Q[s_q_key])
    return Q, policy, episode_rewards


if __name__ == '__main__':
    # 定义要测试的离散化方案列表
    schemes_to_run = ['original', 'scheme1', 'scheme2']
    # schemes_to_run = ['original']

    sarsa_lambda_num_episodes = 10000
    sarsa_lambda_alpha = 0.1
    sarsa_lambda_lambda = 0.9
    trace_implementation = 'replacing'


    all_scheme_results = {}

    for current_scheme_id in schemes_to_run:
        print(f"\n======================================================================")
        print(f"Running SARSA(Lambda) with Discretization Scheme: {current_scheme_id}")
        print(f"Parameters: lambda={sarsa_lambda_lambda}, trace_type={trace_implementation}")
        print(f"======================================================================\n")

        start_time = time.time()
        Q_sarsa_lambda, policy_sarsa_lambda, sarsa_lambda_rewards = sarsa_lambda_algorithm(
            env,  # 使用全局env实例
            num_episodes=sarsa_lambda_num_episodes,
            alpha=sarsa_lambda_alpha,
            lambda_trace=sarsa_lambda_lambda,
            trace_type=trace_implementation,
            scheme_id=current_scheme_id  # <--- 传递当前方案ID
        )
        training_time_sarsa_lambda = time.time() - start_time
        print(
            f"SARSA(Lambda) Training Finished for Scheme: {current_scheme_id}. Time: {training_time_sarsa_lambda:.2f} seconds.")

        q_table_filename = f'q_table_sarsa_lambda_l{sarsa_lambda_lambda}_{trace_implementation}_scheme_{current_scheme_id}.pkl'
        avg_rewards_filename = f'sarsa_lambda_rolling_avg_rewards_l{sarsa_lambda_lambda}_{trace_implementation}_scheme_{current_scheme_id}.npy'
        plot_filename = f'sarsa_lambda_training_rewards_l{sarsa_lambda_lambda}_{trace_implementation}_scheme_{current_scheme_id}.png'

        try:
            with open(q_table_filename, 'wb') as f:
                pickle.dump(dict(Q_sarsa_lambda), f)
            print(f"SARSA(Lambda) Q-table for {current_scheme_id} saved to {q_table_filename}")
        except Exception as e:
            print(f"Error saving Q-table for SARSA(Lambda) scheme {current_scheme_id}: {e}")

        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(1, len(sarsa_lambda_rewards) + 1), sarsa_lambda_rewards,
                 label=f'Raw Rewards (Scheme: {current_scheme_id})')
        rolling_avg_sarsa_lambda_for_plot = []
        if len(sarsa_lambda_rewards) >= 100:
            rolling_avg_sarsa_lambda_for_plot = [np.mean(sarsa_lambda_rewards[max(0, i - 99):i + 1]) for i in
                                                 range(len(sarsa_lambda_rewards))]
            plt.plot(np.arange(1, len(rolling_avg_sarsa_lambda_for_plot) + 1), rolling_avg_sarsa_lambda_for_plot,
                     label=f'Rolling Avg (Scheme: {current_scheme_id})', color='green', alpha=0.7)
            np.save(avg_rewards_filename, np.array(rolling_avg_sarsa_lambda_for_plot))
            print(f"SARSA(Lambda) rolling average rewards for {current_scheme_id} saved to {avg_rewards_filename}")

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(
            f'SARSA(Lambda={sarsa_lambda_lambda}, {trace_implementation}, Scheme: {current_scheme_id}) Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_filename)
        plt.close()

        # --- 最终策略评估 ---
        test_episodes = 100
        total_rewards_sarsa_lambda_test = []
        landed_successfully_sarsa_lambda = 0
        fuel_consumption_sarsa_lambda_test = []

        print(f"\nTesting SARSA(Lambda) Policy for Scheme: {current_scheme_id}...")
        for i_test_episode in range(test_episodes):
            observation, info = env.reset()
            discretized_s = discretize_state(observation, scheme_id=current_scheme_id)  # <--- 测试时也使用对应方案
            terminated = False
            truncated = False
            episode_reward = 0
            episode_fuel_estimate = 0.0
            for t_step in range(1000):
                action = policy_sarsa_lambda.get(discretized_s, env.action_space.sample())
                if action == 2:
                    episode_fuel_estimate += 0.3
                elif action == 1 or action == 3:
                    episode_fuel_estimate += 0.03
                next_observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                discretized_s = discretize_state(next_observation, scheme_id=current_scheme_id)  # <--- 测试时也使用对应方案
                if terminated or truncated: break
            total_rewards_sarsa_lambda_test.append(episode_reward)
            fuel_consumption_sarsa_lambda_test.append(episode_fuel_estimate)
            if episode_reward > 200: landed_successfully_sarsa_lambda += 1

        avg_reward = np.mean(total_rewards_sarsa_lambda_test) if total_rewards_sarsa_lambda_test else -float('inf')
        success_rate = (landed_successfully_sarsa_lambda / test_episodes) * 100 if test_episodes > 0 else 0
        avg_fuel = np.mean(fuel_consumption_sarsa_lambda_test) if fuel_consumption_sarsa_lambda_test else 0

        print(f"\n--- SARSA(Lambda) Final Policy Evaluation (Scheme: {current_scheme_id}) ---")
        print(f"Average Test Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Average Fuel Consumption (estimate): {avg_fuel:.2f} (points penalty based)")
        print(f"Total Training Time: {training_time_sarsa_lambda:.2f} seconds")

        # (可选) 存储每个方案的评估结果
        all_scheme_results[current_scheme_id] = {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_fuel': avg_fuel,
            'training_time': training_time_sarsa_lambda
        }

    print("\n\n--- Summary of All Scheme Results ---")
    for scheme, results in all_scheme_results.items():
        print(
            f"Scheme: {scheme} -> Avg Reward: {results['avg_reward']:.2f}, Success: {results['success_rate']:.2f}%, Fuel: {results['avg_fuel']:.2f}, Time: {results['training_time']:.2f}s")

    env.close()