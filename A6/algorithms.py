import numpy as np
import random
from collections import defaultdict
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_episode(env, policy):
    """
    Generates an episode following a given policy.
    The policy here is implicitly the random policy defined in the project (0.25 for each action).
    Args:
        env: The GridWorld environment object.
        policy: A function that takes a state and returns an action.
                For this project, it's a random choice among possible actions.
    Returns:
        A list of (state, action, reward) tuples.
        States are represented as (row, col) tuples.
    """
    episode = []
    current_state_pos = env.reset()
    done = False
    max_episode_length = 200
    count = 0

    while not done and count < max_episode_length:
        action_index = random.choice(range(env.num_actions))
        next_state_pos, reward, done = env.step(action_index)
        episode.append((current_state_pos, action_index, reward))
        current_state_pos = next_state_pos
        count += 1
    return episode


def monte_carlo_prediction(env, num_episodes, gamma=0.99, mode='first-visit'):
    """
    Monte Carlo Prediction algorithm for estimating V(s).
    Args:
        env: The GridWorld environment.
        num_episodes: Number of episodes to generate.
        gamma: Discount factor.
        mode: 'first-visit' or 'every-visit'.
    Returns:
        V: A dictionary mapping state_index to its estimated value.
        V_history: A list of V dictionaries at certain episode intervals (for plotting convergence)
    """
    V = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for r in range(env.size):
        for c in range(env.size):
            if not env.is_obstacle(r, c):
                V[env.get_state_index((r, c))] = 0.0

    V_history_interval = max(1, num_episodes // 20)
    V_history = []

    for i in range(num_episodes):
        episode = generate_episode(env, None)
        G = 0
        states_in_episode_for_fv = [env.get_state_index(s_a_r[0]) for s_a_r in episode]

        for t in range(len(episode) - 1, -1, -1):
            state_pos, _, reward = episode[t]
            state_index = env.get_state_index(state_pos)
            G = gamma * G + reward

            if mode == 'first-visit':
                is_first_visit_at_t = True
                for k in range(t):
                    if states_in_episode_for_fv[k] == state_index:
                        is_first_visit_at_t = False
                        break
                if is_first_visit_at_t:
                    returns_sum[state_index] += G
                    returns_count[state_index] += 1
                    if returns_count[state_index] > 0:
                        V[state_index] = returns_sum[state_index] / returns_count[state_index]
            elif mode == 'every-visit':
                returns_sum[state_index] += G
                returns_count[state_index] += 1
                if returns_count[state_index] > 0:
                    V[state_index] = returns_sum[state_index] / returns_count[state_index]

        if (i + 1) % V_history_interval == 0 or i == num_episodes - 1:
            V_hist_copy = defaultdict(float)
            for r_idx in range(env.size):
                for c_idx in range(env.size):
                    s_rc_h = (r_idx, c_idx)
                    s_idx_h = env.get_state_index(s_rc_h)
                    if not env.is_obstacle(r_idx, c_idx):
                        V_hist_copy[s_idx_h] = V.get(s_idx_h, 0.0)
            V_history.append(V_hist_copy)

    final_V = defaultdict(float)
    for r_idx in range(env.size):
        for c_idx in range(env.size):
            state_rc = (r_idx, c_idx)
            s_idx = env.get_state_index(state_rc)
            if env.is_terminal_state[state_rc]:
                final_V[s_idx] = 0.0
            elif env.is_obstacle(r_idx, c_idx):
                final_V[s_idx] = 0.0
            else:
                final_V[s_idx] = V.get(s_idx, 0.0)
    return final_V, V_history


def td0_prediction(env, num_episodes, alpha, gamma=0.99):
    """
    Temporal Difference (TD(0)) Prediction algorithm for estimating V(s).
    Args:
        env: The GridWorld environment.
        num_episodes: Number of episodes to run.
        alpha: Learning rate.
        gamma: Discount factor.
    Returns:
        V: A dictionary mapping state_index to its estimated value.
        V_history: A list of V dictionaries at certain episode intervals.
    """
    V = defaultdict(float)
    for r in range(env.size):
        for c in range(env.size):
            state_rc = (r, c)
            s_idx = env.get_state_index(state_rc)
            if not env.is_obstacle(r, c):
                if env.is_terminal_state[state_rc]:
                    V[s_idx] = 0.0  # Value of terminal state is 0
                else:
                    V[s_idx] = 0.0

    V_history_interval = max(1, num_episodes // 20)
    V_history = []
    max_episode_length = 200

    for i in range(num_episodes):
        current_state_pos = env.reset()
        done = False
        count = 0
        while not done and count < max_episode_length:
            current_state_idx = env.get_state_index(current_state_pos)
            action_index = random.choice(range(env.num_actions))
            next_state_pos, reward, done = env.step(action_index)
            next_state_idx = env.get_state_index(next_state_pos)

            v_next_state = 0.0 if env.is_terminal_state[next_state_pos] else V[next_state_idx]
            td_target = reward + gamma * v_next_state
            td_error = td_target - V[current_state_idx]

            if not env.is_terminal_state[current_state_pos]:  # Only update non-terminal states
                V[current_state_idx] = V[current_state_idx] + alpha * td_error

            current_state_pos = next_state_pos
            count += 1

        if (i + 1) % V_history_interval == 0 or i == num_episodes - 1:
            V_copy = defaultdict(float)
            for r_idx in range(env.size):
                for c_idx in range(env.size):
                    state_rc_hist = (r_idx, c_idx)
                    s_idx_hist = env.get_state_index(state_rc_hist)
                    if not env.is_obstacle(r_idx, c_idx):
                        V_copy[s_idx_hist] = V.get(s_idx_hist, 0.0)
            V_history.append(V_copy)

    final_V = defaultdict(float)
    for r_idx in range(env.size):
        for c_idx in range(env.size):
            state_rc = (r_idx, c_idx)
            s_idx = env.get_state_index(state_rc)
            if env.is_terminal_state[state_rc]:
                final_V[s_idx] = 0.0
            elif env.is_obstacle(r_idx, c_idx):
                final_V[s_idx] = 0.0
            else:
                final_V[s_idx] = V.get(s_idx, 0.0)
    return final_V, V_history


# --- New TD(lambda) Algorithms ---

def td_lambda_forward_view(env, num_episodes, alpha, lambda_param, gamma=0.99):
    """
    Forward View TD(lambda) Prediction. This is an offline update.
    Args:
        env: The GridWorld environment.
        num_episodes: Number of episodes to run.
        alpha: Learning rate (can be seen as step size for each G_lambda_t).
        lambda_param: Lambda for TD(lambda).
        gamma: Discount factor.
    Returns:
        V: A dictionary mapping state_index to its estimated value.
        V_history: A list of V dictionaries for convergence plotting.
    """
    V = defaultdict(float)
    for r in range(env.size):
        for c in range(env.size):
            state_rc = (r, c)
            s_idx = env.get_state_index(state_rc)
            if not env.is_obstacle(r, c): V[s_idx] = 0.0

    V_history_interval = max(1, num_episodes // 20)
    V_history = []

    for i in range(num_episodes):
        episode = generate_episode(env, None)
        T = len(episode)

        for t in range(T):
            current_state_pos, _, _ = episode[t]
            current_state_idx = env.get_state_index(current_state_pos)

            if env.is_terminal_state[current_state_pos]:
                continue

            g_lambda_t = 0.0
            if lambda_param == 1.0:  # Equivalent to Monte Carlo
                current_G_t = 0.0
                for k_idx_ep in range(t, T):
                    reward_k_plus_1 = episode[k_idx_ep][2]
                    current_G_t += (gamma ** (k_idx_ep - t)) * reward_k_plus_1
                g_lambda_t = current_G_t
            else:
                for n in range(1, T - t):
                    current_g_n_step = 0.0
                    rewards_path = [ep[2] for ep in episode[t: t + n]]  # R_{t+1} ... R_{t+n}
                    for k_idx, r_val in enumerate(rewards_path):
                        current_g_n_step += (gamma ** k_idx) * r_val

                    s_t_n_plus_1_pos = episode[t + n][0]  # State S_{t+n}
                    v_s_t_n_plus_1 = 0.0 if env.is_terminal_state[s_t_n_plus_1_pos] else V[
                        env.get_state_index(s_t_n_plus_1_pos)]
                    current_g_n_step += (gamma ** n) * v_s_t_n_plus_1

                    g_lambda_t += (1 - lambda_param) * (lambda_param ** (n - 1)) * current_g_n_step

                if T - t > 0:  # If there's at least one step remaining to form G_t
                    actual_G_t = 0.0
                    for k_idx_ep in range(t, T):
                        reward_k_plus_1 = episode[k_idx_ep][2]
                        actual_G_t += (gamma ** (k_idx_ep - t)) * reward_k_plus_1

                    if T - t - 1 >= 0:  # Check if power is non-negative
                        g_lambda_t += (lambda_param ** (T - t - 1)) * actual_G_t

            V[current_state_idx] += alpha * (g_lambda_t - V[current_state_idx])

        if (i + 1) % V_history_interval == 0 or i == num_episodes - 1:
            V_copy = defaultdict(float)
            for r_idx in range(env.size):
                for c_idx in range(env.size):
                    s_rc_h = (r_idx, c_idx)
                    s_idx_h = env.get_state_index(s_rc_h)
                    if not env.is_obstacle(r_idx, c_idx): V_copy[s_idx_h] = V.get(s_idx_h, 0.0)
            V_history.append(V_copy)

    final_V = defaultdict(float)
    for r_idx in range(env.size):
        for c_idx in range(env.size):
            state_rc = (r_idx, c_idx)
            s_idx = env.get_state_index(state_rc)
            if not env.is_obstacle(r_idx, c_idx): final_V[s_idx] = V.get(s_idx, 0.0)
            if env.is_terminal_state[state_rc] or env.is_obstacle(r_idx, c_idx):
                final_V[s_idx] = 0.0
            else:
                final_V[s_idx] = V.get(s_idx, 0.0)
    return final_V, V_history


def td_lambda_backward_view(env, num_episodes, alpha, lambda_param, gamma=0.99, trace_type='replacing'):
    """
    Backward View TD(lambda) Prediction with Eligibility Traces. This is an online update.
    Args:
        env: The GridWorld environment.
        num_episodes: Number of episodes to run.
        alpha: Learning rate.
        lambda_param: Lambda for eligibility traces.
        gamma: Discount factor.
        trace_type: 'accumulating' or 'replacing'.
    Returns:
        V: A dictionary mapping state_index to its estimated value.
        V_history: A list of V dictionaries for convergence plotting.
        eligibility_traces_history: List of E dictionaries for visualization (optional)
    """
    V = defaultdict(float)
    for r in range(env.size):
        for c in range(env.size):
            state_rc = (r, c)
            s_idx = env.get_state_index(state_rc)
            if not env.is_obstacle(r, c): V[s_idx] = 0.0
            # Terminal states also initialized to 0.0 implicitly by V's init or above.

    V_history_interval = max(1, num_episodes // 20)
    V_history = []
    eligibility_traces_history = []
    max_episode_length = 200

    for i in range(num_episodes):
        E = defaultdict(float)  # Reset eligibility traces for each new episode
        current_state_pos = env.reset()
        done = False
        count = 0
        episode_E_snapshots = []

        while not done and count < max_episode_length:
            current_state_idx = env.get_state_index(current_state_pos)

            if trace_type == 'replacing':
                E[current_state_idx] = 1.0
            elif trace_type == 'accumulating':
                E[current_state_idx] += 1.0

            if lambda_param > 0 and lambda_param < 1:
                episode_E_snapshots.append(dict(E))  # Store E for visualization

            action_index = random.choice(range(env.num_actions))
            next_state_pos, reward, done = env.step(action_index)
            next_state_idx = env.get_state_index(next_state_pos)

            v_s = V[current_state_idx]
            v_s_prime = 0.0 if env.is_terminal_state[next_state_pos] else V[next_state_idx]
            td_error = reward + gamma * v_s_prime - v_s

            states_with_trace = list(E.keys())  # Iterate over states that have a trace
            for s_idx_trace in states_with_trace:
                if not env.is_terminal_state[env.get_pos_from_index(s_idx_trace)]:
                    V[s_idx_trace] += alpha * td_error * E[s_idx_trace]

                E[s_idx_trace] *= gamma * lambda_param
                if E[s_idx_trace] < 1e-6:  # Prune small traces
                    del E[s_idx_trace]

            current_state_pos = next_state_pos
            count += 1

        if (i + 1) % V_history_interval == 0 or i == num_episodes - 1:
            V_copy = defaultdict(float)
            for r_idx in range(env.size):
                for c_idx in range(env.size):
                    s_rc_h = (r_idx, c_idx)
                    s_idx_h = env.get_state_index(s_rc_h)
                    if not env.is_obstacle(r_idx, c_idx): V_copy[s_idx_h] = V.get(s_idx_h, 0.0)
            V_history.append(V_copy)
            if episode_E_snapshots and (lambda_param == 0.5):  # Store E history for lambda=0.5
                eligibility_traces_history.append(list(episode_E_snapshots))

    final_V = defaultdict(float)
    for r_idx in range(env.size):
        for c_idx in range(env.size):
            state_rc = (r_idx, c_idx)
            s_idx = env.get_state_index(state_rc)
            if env.is_terminal_state[state_rc] or env.is_obstacle(r_idx, c_idx):
                final_V[s_idx] = 0.0
            else:
                final_V[s_idx] = V.get(s_idx, 0.0)

    return final_V, V_history, eligibility_traces_history


if __name__ == '__main__':
    class DummyEnv:
        def __init__(self, size=5):
            self.size = size
            self.rewards = np.zeros((size, size))
            self.is_terminal_state = np.full((size, size), False, dtype=bool)
            self.actions = ['up', 'down', 'left', 'right']
            self.num_actions = len(self.actions)
            self.obstacles_pos = []
            self.goal_pos = (size - 1, size - 1)
            self.rewards[self.goal_pos] = 10
            self.is_terminal_state[self.goal_pos] = True
            self.traps_pos = [(1, 1)]
            for trap in self.traps_pos:
                self.rewards[trap] = -10
                self.is_terminal_state[trap] = True
            self.current_pos = (0, 0)

        def reset(self):
            self.current_pos = (0, 0)
            return self.current_pos

        def step(self, action_index):
            r, c = self.current_pos
            action = self.actions[action_index]
            if action == 'up':
                r = max(0, r - 1)
            elif action == 'down':
                r = min(self.size - 1, r + 1)
            elif action == 'left':
                c = max(0, c - 1)
            elif action == 'right':
                c = min(self.size - 1, c + 1)
            next_pos = (r, c) if not (r, c) in self.obstacles_pos else self.current_pos
            self.current_pos = next_pos
            reward = self.rewards[next_pos]
            done = self.is_terminal_state[next_pos]
            return next_pos, reward, done

        def get_state_index(self, state_pos):
            return state_pos[0] * self.size + state_pos[1]

        def get_pos_from_index(self, state_idx):
            return state_idx // self.size, state_idx % self.size

        def is_obstacle(self, r, c):
            return (r, c) in self.obstacles_pos

        def get_all_states(self):
            return [(r, c) for r in range(self.size) for c in range(self.size) if not self.is_obstacle(r, c)]


    print("Testing algorithms.py with TD(lambda)...")
    test_env = DummyEnv(size=5)
    test_env.obstacles_pos = [(1, 2), (2, 1), (3, 3)]

    num_episodes_test = 500
    gamma_test = 0.99
    alpha_test = 0.1
    lambda_test_values = [0.0, 0.5, 1.0]

    print("\nRunning First-Visit Monte Carlo Prediction...")
    V_mc_first, hist_mc_first = monte_carlo_prediction(test_env, num_episodes_test, gamma_test, mode='first-visit')
    print(f"V_mc_first (State 0): {V_mc_first.get(0, 0.0):.2f}")

    print("\nRunning Every-Visit Monte Carlo Prediction...")
    V_mc_every, hist_mc_every = monte_carlo_prediction(test_env, num_episodes_test, gamma_test, mode='every-visit')
    print(f"V_mc_every (State 0): {V_mc_every.get(0, 0.0):.2f}")

    print("\nRunning TD(0) Prediction...")
    V_td0, hist_td0 = td0_prediction(test_env, num_episodes_test, alpha_test, gamma_test)
    print(f"V_td0 (State 0): {V_td0.get(0, 0.0):.2f}")

    for lmbda in lambda_test_values:
        print(f"\nRunning Forward View TD(lambda={lmbda})...")
        V_td_fwd, hist_td_fwd = td_lambda_forward_view(test_env, num_episodes_test, alpha_test, lmbda, gamma_test)
        print(f"V_td_fwd (lambda={lmbda}, State 0): {V_td_fwd.get(0, 0.0):.2f}")

    for lmbda in lambda_test_values:
        print(f"\nRunning Backward View TD(lambda={lmbda}) with replacing traces...")
        V_td_bwd, hist_td_bwd, E_hist_bwd = td_lambda_backward_view(test_env, num_episodes_test, alpha_test, lmbda,
                                                                    gamma_test, trace_type='replacing')
        print(f"V_td_bwd (lambda={lmbda}, State 0): {V_td_bwd.get(0, 0.0):.2f}")

        print(f"\nRunning Backward View TD(lambda={lmbda}) with accumulating traces...")
        V_td_bwd_acc, hist_td_bwd_acc, E_hist_bwd_acc = td_lambda_backward_view(test_env, num_episodes_test, alpha_test,
                                                                                lmbda, gamma_test,
                                                                                trace_type='accumulating')
        print(f"V_td_bwd_acc (lambda={lmbda}, State 0): {V_td_bwd_acc.get(0, 0.0):.2f}")

    print("\nAll algorithm test runs completed.")