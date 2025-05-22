import gym
import numpy as np
import pickle
import random
from collections import defaultdict
import time

try:
    from lunar_lander_utils import discretize_state
except ImportError:
    print("错误：无法导入 discretize_state 函数。")
    print("请确保 lunar_lander_utils.py 文件与此脚本在同一目录，或者在PYTHONPATH中。")
    exit()

Q_TABLE_PATH = 'q_table_q_learning.pkl'
NUM_EPISODES_TO_RENDER = 3
MAX_STEPS_PER_EPISODE = 1000


def load_q_learning_policy(q_table_path, env_action_space_n):
    """从保存的Q表加载贪婪策略"""
    try:
        with open(q_table_path, 'rb') as f:
            # pickle加载的是保存时的dict类型
            q_table_loaded_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: Q表文件 {q_table_path} 未找到。")
        return None
    except Exception as e:
        print(f"加载Q表 {q_table_path} 时出错: {e}")
        return None

    # 从加载的Q表（字典）构建策略
    # 对于测试中可能遇到的、训练时未见过的状态，策略返回一个随机动作
    policy = defaultdict(lambda: random.randrange(env_action_space_n))
    for state_key, action_values in q_table_loaded_dict.items():
        policy[state_key] = np.argmax(action_values)

    print(f"成功从 {q_table_path} 加载Q表并构建策略。Q表包含 {len(q_table_loaded_dict)} 个状态。")
    return policy


if __name__ == '__main__':
    try:
        env = gym.make('LunarLander-v2', render_mode='human')
    except Exception as e:
        print(f"创建渲染环境失败: {e}")
        print("请确保您已正确安装了 'box2d-py' 和 'pygame'。") # 依赖说明
        exit()

    action_size = env.action_space.n
    policy_q_learning = load_q_learning_policy(Q_TABLE_PATH, action_size)

    if policy_q_learning is None:
        print("无法加载Q-learning策略，程序退出。")
        env.close()
        exit()

    print(f"\n开始渲染Q-learning策略，共 {NUM_EPISODES_TO_RENDER} 个回合...")
    for i_episode in range(1, NUM_EPISODES_TO_RENDER + 1):
        observation, info = env.reset()
        discretized_state = discretize_state(observation)  # Q-learning使用离散化状态

        episode_reward = 0
        terminated = False
        truncated = False

        print(f"\n--- 开始渲染第 {i_episode}/{NUM_EPISODES_TO_RENDER} 回合 (Q-learning) ---")
        # 此时可以开始您的屏幕录制软件

        for t_step in range(MAX_STEPS_PER_EPISODE):
            # 对于 'human' 模式，通常不需要在循环中显式调用 env.render()

            # 从加载的策略中获取动作
            action = policy_q_learning.get(discretized_state, env.action_space.sample())

            next_observation, reward, terminated, truncated, info = env.step(action)

            discretized_state = discretize_state(next_observation) # 更新离散化后的状态
            episode_reward += reward

            if terminated or truncated:
                print(f"第 {i_episode} 回合结束于 {t_step + 1} 步。最终奖励: {episode_reward:.2f}")
                break

        if not (terminated or truncated): # 如果是因为达到最大步数而结束
            print(f"第 {i_episode} 回合达到最大步数 {MAX_STEPS_PER_EPISODE}。最终奖励: {episode_reward:.2f}")

        if i_episode < NUM_EPISODES_TO_RENDER:
            print("准备开始下一回合...")
            time.sleep(2)  # 暂停2秒，方便观看和录制切换

    print("\n--- 所有Q-learning渲染回合结束 ---")
    env.close()