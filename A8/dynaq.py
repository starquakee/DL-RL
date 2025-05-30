import numpy as np
from collections import defaultdict
import random


class Discretizer:
    def __init__(self, num_bins_per_dim, state_lows, state_highs):
        """
        初始化状态离散化器。
        参数:
            num_bins_per_dim: list/tuple, 每个状态维度要划分的格子数。
            state_lows: list/tuple, 每个状态维度的下界。
            state_highs: list/tuple, 每个状态维度的上界。
        """
        if not (len(num_bins_per_dim) == len(state_lows) == len(state_highs)):
            raise ValueError("Mismatch in lengths of discretization parameters.")

        self.num_bins_per_dim = num_bins_per_dim
        self.state_lows = np.array(state_lows)
        self.state_highs = np.array(state_highs)
        self.bin_edges = []

        for i in range(len(self.num_bins_per_dim)):
            # 对于布尔型/已离散的维度 (例如只有0和1)，确保bins能正确映射
            if self.state_lows[i] == 0.0 and self.state_highs[i] == 1.0 and self.num_bins_per_dim[i] == 2:
                # 特殊处理，使得0映射到bin 0, 1映射到bin 1
                edges = np.array([0.5])
            else:
                edges = np.linspace(self.state_lows[i], self.state_highs[i],
                                    num=self.num_bins_per_dim[i] - 1, endpoint=False)
                # 移除那些可能因为浮点数问题而超出上界的边缘 (np.digitize的特性)
                # 我们只关心内部的 N-1 个分割点
            self.bin_edges.append(edges)

    def discretize(self, state_continuous):
        """
        将连续状态向量转换为离散状态索引的元组。
        """
        discrete_indices = []
        for i in range(len(state_continuous)):
            # np.digitize 返回的是值应该插入到哪个bin的索引 (从1开始计数)
            # 我们希望索引从0开始，所以减1
            # clip确保值在定义的 low/high 范围内，防止 digitize 出错或给出意外索引
            clipped_val = np.clip(state_continuous[i], self.state_lows[i], self.state_highs[i])

            # 对于已经明确只有两个值的维度 (0 or 1)
            if self.state_lows[i] == 0.0 and self.state_highs[i] == 1.0 and self.num_bins_per_dim[i] == 2:
                idx = 0 if clipped_val < 0.5 else 1
            else:
                idx = np.digitize(clipped_val, self.bin_edges[i], right=False)

            discrete_indices.append(idx)
        return tuple(discrete_indices)



class DynaQ:
    def __init__(self, state_dim, action_dim, bins_per_dim, state_lows, state_highs,
                 alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay_episodes=1000,
                 planning_steps=5):

        self.action_dim = action_dim
        self.discretizer = Discretizer(bins_per_dim, state_lows, state_highs)

        # Q-table 初始化为零
        q_table_shape = tuple(bins_per_dim) + (action_dim,)
        self.q_table = np.zeros(q_table_shape)

        # 环境模型: model[(discrete_state, action)] = (reward, next_discrete_state, done)
        self.model = {}
        self.observed_state_action_pairs = set()  # 用于从已访问的(s,a)中采样

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon_start  # 当前探索率
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_value = (
                        epsilon_start - epsilon_min) / epsilon_decay_episodes if epsilon_decay_episodes > 0 else 0

        self.planning_steps = planning_steps

    def get_discrete_state(self, state_continuous):
        """ 连续状态离散化 """
        return self.discretizer.discretize(state_continuous)

    def choose_action(self, state_continuous, training=True):
        """ epsilon-贪婪策略选择动作 """
        discrete_state = self.get_discrete_state(state_continuous)

        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 随机探索
        else:
            q_values = self.q_table[discrete_state]
            max_q = np.max(q_values)
            # 找出所有具有最大Q值的动作的索引
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions)  # 从最佳动作中随机选择一个

    def decay_epsilon(self):
        """ 在每个episode结束后调用，用于衰减epsilon """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_value
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def update(self, s_continuous, action, reward, s_next_continuous, done):
        """ 真实经验学习 + 模型学习 + 模型规划 """
        s_discrete = self.get_discrete_state(s_continuous)
        s_next_discrete = self.get_discrete_state(s_next_continuous)

        # 1. 直接 Q-Learning 更新 (真实经验)
        current_q_value = self.q_table[s_discrete][action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[s_next_discrete])

        self.q_table[s_discrete][action] += self.alpha * (td_target - current_q_value)

        # 2. 模型学习
        # 将经验 (s, a) -> (r, s') 存储到模型中
        # (s_discrete, action) 必须是可哈希的，元组可以
        model_key = (s_discrete, action)
        self.model[model_key] = (reward, s_next_discrete, done)
        self.observed_state_action_pairs.add(model_key)  # 记录已访问的(s,a)对

        # 3. 规划 (从模型中学习)
        self._planning()

    def _planning(self):
        """ 从模型随机采样进行规划 """
        if not self.observed_state_action_pairs:  # 如果模型为空，则无法规划
            return

        for _ in range(self.planning_steps):
            sampled_s_discrete, sampled_action = random.choice(list(self.observed_state_action_pairs))
            if (sampled_s_discrete, sampled_action) not in self.model:
                continue  # 跳过这个规划步骤

            r_model, s_next_model_discrete, done_model = self.model[(sampled_s_discrete, sampled_action)]

            # 使用模拟经验进行 Q-Learning 更新
            current_q_model = self.q_table[sampled_s_discrete][sampled_action]
            if done_model:
                td_target_model = r_model
            else:
                td_target_model = r_model + self.gamma * np.max(self.q_table[s_next_model_discrete])

            self.q_table[sampled_s_discrete][sampled_action] += self.alpha * (td_target_model - current_q_model)