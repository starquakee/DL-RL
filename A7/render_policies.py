import gym
import numpy as np
import pickle
import torch
import random
from collections import defaultdict

try:
    from lunar_lander_utils import discretize_state
    from dqn_lander import DQNAgent, QNetwork, device  # device 也很重要
except ImportError as e:
    print(f"无法导入必要的模块，请确保 lunar_lander_utils.py 和 dqn_lander.py 文件存在且包含所需定义: {e}")
    exit()


def load_tabular_policy(q_table_path, env_action_space_n):
    """从保存的Q表加载贪婪策略"""
    try:
        with open(q_table_path, 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: Q表文件 {q_table_path} 未找到。")
        return None
    except Exception as e:
        print(f"加载Q表时出错 {q_table_path}: {e}")
        return None

    policy = defaultdict(lambda: random.randrange(env_action_space_n))
    for state_key, action_values in q_table.items():
        policy[state_key] = np.argmax(action_values)
    return policy


def run_and_render(env, policy_or_agent, algorithm_name, num_episodes=3, is_dqn=False):
    """加载策略/模型并渲染几个回合"""
    print(f"\n--- Rendering: {algorithm_name} ---")
    if policy_or_agent is None:
        print(f"无法为 {algorithm_name} 加载策略/模型，跳过渲染。")
        return

    for i_episode in range(1, num_episodes + 1):
        if is_dqn:
            state, info = env.reset()
        else:  # Tabular
            observation, info = env.reset()
            state = discretize_state(observation)  # 对于表格方法，状态是离散化的

        print(f"  {algorithm_name} - Episode {i_episode}")
        terminated = False
        truncated = False
        episode_reward = 0

        for t_step in range(1000):

            if is_dqn:
                action = policy_or_agent.act(state, eps=0.0)  # DQN使用贪婪策略
            else:
                action = policy_or_agent.get(state, env.action_space.sample())  # 从策略字典获取动作

            next_state_raw, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if is_dqn:
                state = next_state_raw
            else:
                state = discretize_state(next_state_raw)

            if terminated or truncated:
                print(f"  Episode finished after {t_step + 1} steps. Reward: {episode_reward:.2f}")
                break
        if not (terminated or truncated):
            print(f"  Episode reached max steps. Reward: {episode_reward:.2f}")
    print(f"--- Finished Rendering {algorithm_name} ---")


if __name__ == '__main__':
    try:
        env_render = gym.make('LunarLander-v2', render_mode='human')
    except Exception as e:
        print(f"创建渲染环境失败: {e}")
        print("请确保您已安装了 'box2d-py' 和 'pygame'。")
        exit()

    action_size = env_render.action_space.n
    state_size = env_render.observation_space.shape[0]  # 主要为DQN agent初始化

    # --- 渲染 Monte Carlo ---
    policy_mc = load_tabular_policy('q_table_mc.pkl', action_size)
    run_and_render(env_render, policy_mc, "Monte Carlo", is_dqn=False)
    input("按 Enter键 继续渲染下一个算法...")  # 等待用户确认，方便观看

    # --- 渲染 SARSA ---
    policy_sarsa = load_tabular_policy('q_table_sarsa.pkl', action_size)
    run_and_render(env_render, policy_sarsa, "SARSA", is_dqn=False)
    input("按 Enter键 继续渲染下一个算法...")

    # --- 渲染 SARSA(Lambda) ---
    policy_sarsa_lambda = load_tabular_policy('q_table_sarsa_lambda.pkl', action_size)  # 修改为您实际的文件名
    run_and_render(env_render, policy_sarsa_lambda, "SARSA(Lambda)", is_dqn=False)
    input("按 Enter键 继续渲染下一个算法...")

    # --- 渲染 Q-learning ---
    policy_q_learning = load_tabular_policy('q_table_q_learning.pkl', action_size)
    run_and_render(env_render, policy_q_learning, "Q-learning", is_dqn=False)
    input("按 Enter键 继续渲染下一个算法...")

    # --- 渲染 DQN ---
    try:
        dqn_agent_render = DQNAgent(state_size=state_size, action_size=action_size, seed=0)  # 确保seed与训练时一致或不重要
        dqn_agent_render.qnetwork_local.load_state_dict(torch.load('dqn_lander_weights.pth', map_location=device))
        dqn_agent_render.qnetwork_local.eval()  # 设置为评估模式
        run_and_render(env_render, dqn_agent_render, "DQN", is_dqn=True)
    except FileNotFoundError:
        print("错误: DQN权重文件 'dqn_lander_weights.pth' 未找到。")
    except Exception as e:
        print(f"加载或渲染DQN时出错: {e}")

    print("所有算法渲染完毕。")
    env_render.close()
