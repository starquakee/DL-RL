import gym
import torch
import numpy as np
import time
try:
    from dqn_lander import DQNAgent, QNetwork, device
except ImportError:
    print("错误：无法导入 DQNAgent 或 QNetwork。")
    print("请确保 dqn_lander.py 文件与此脚本在同一目录，或者在PYTHONPATH中。")
    print("并且该文件中包含了 DQNAgent 和 QNetwork 的定义以及 device 的设置。")
    exit()

DQN_WEIGHTS_PATH = 'dqn_lander_weights.pth'
NUM_EPISODES_TO_RENDER = 3
MAX_STEPS_PER_EPISODE = 1000

if __name__ == '__main__':
    try:
        env = gym.make('LunarLander-v2', render_mode='human')
    except Exception as e:
        print(f"创建渲染环境失败: {e}")
        print("请确保您已正确安装了 'box2d-py' 和 'pygame'。") # 依赖说明
        exit()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 初始化DQN Agent
    # seed=0 是之前代码中常用的，您可以根据需要调整或移除。
    # 网络结构相关的参数(state_size, action_size, fc_units in QNetwork)需与训练时匹配。
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)

    # 加载训练好的模型权重
    try:
        # map_location=device 确保模型加载到正确的设备 (CPU/GPU)
        agent.qnetwork_local.load_state_dict(torch.load(DQN_WEIGHTS_PATH, map_location=device))
        print(f"成功从 {DQN_WEIGHTS_PATH} 加载模型权重。")
    except FileNotFoundError:
        print(f"错误: 未找到DQN权重文件 '{DQN_WEIGHTS_PATH}'。请确保文件名和路径正确。")
        env.close()
        exit()
    except Exception as e:
        print(f"加载DQN模型权重时发生错误: {e}")
        env.close()
        exit()

    # 设置网络为评估模式 (这会关闭dropout等层)
    agent.qnetwork_local.eval()

    print(f"\n开始渲染DQN策略，共 {NUM_EPISODES_TO_RENDER} 个回合...")
    for i_episode in range(1, NUM_EPISODES_TO_RENDER + 1):
        state, info = env.reset()  # DQN使用原始状态 (numpy array)
        episode_reward = 0
        terminated = False
        truncated = False

        print(f"\n--- 开始渲染第 {i_episode}/{NUM_EPISODES_TO_RENDER} 回合 (DQN) ---")
        # 此时可以开始您的屏幕录制软件

        for t_step in range(MAX_STEPS_PER_EPISODE):
            # 对于 'human' 模式，通常不需要在循环中显式调用 env.render()

            # 使用贪婪策略选择动作 (epsilon = 0)
            action = agent.act(state, eps=0.0)

            next_state, reward, terminated, truncated, info = env.step(action)

            state = next_state # 更新状态
            episode_reward += reward

            if terminated or truncated:
                print(f"第 {i_episode} 回合结束于 {t_step + 1} 步。最终奖励: {episode_reward:.2f}")
                break

        if not (terminated or truncated): # 如果是因为达到最大步数而结束
            print(f"第 {i_episode} 回合达到最大步数 {MAX_STEPS_PER_EPISODE}。最终奖励: {episode_reward:.2f}")

        if i_episode < NUM_EPISODES_TO_RENDER:
            print("准备开始下一回合...")
            time.sleep(2)  # 暂停2秒，方便观看和录制切换

    print("\n--- 所有DQN渲染回合结束 ---")
    env.close()