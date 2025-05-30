# import gymnasium as gym
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque  # For moving average of training rewards
from reinforce import REINFORCE
from qac import QAC
from aac import A2C
from nac import NaturalAC
from dynaq import DynaQ
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from collections import defaultdict


# --- Hyperparameters and Setup ---
ENV_NAME = "LunarLander-v2"
NUM_EPISODES = 2000
EVAL_FREQ = 100
NUM_TEST_EPISODES = 10
MAX_STEPS_PER_EPISODE = 1000  # Max steps for LunarLander before truncation

# Common params
GAMMA = 0.99
# Learning rates
LR_NN = 0.001  # For REINFORCE, QAC actor/critic, A2C actor/critic, NaturalAC critic
LR_DYNQ_ALPHA = 0.1  # Alpha for DynaQ's Q-table updates

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_EPISODES = 1000

# Specific A2C/NaturalAC params
GAE_LAMBDA = 0.95
# Specific NaturalAC params
CG_ITERS = 10
CG_DAMPING = 1e-3
MAX_KL_STEP = 0.01
FIM_UPDATE_INTERVAL = 1  # Update FIM at every main update call

# DynaQ params
PLANNING_STEPS = 5

DYNQ_BINS = [10, 10, 8, 8, 6, 6, 2, 2]
# Approximate state bounds for LunarLander-v2 (can be tuned)
# Based on typical ranges rather than -inf, inf
STATE_LOWS = np.array([-1.2, -0.3, -2.0, -2.0, -np.pi / 2, -2.0, 0.0, 0.0])  # angle restricted
STATE_HIGHS = np.array([1.2, 1.5, 2.0, 2.0, np.pi / 2, 2.0, 1.0, 1.0])


# --- Evaluation Function ---
def evaluate_agent(env, agent, num_episodes=10):
    total_rewards = []
    total_successes = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        while not done and not truncated:
            if isinstance(agent, REINFORCE):
                action, _ = agent.select_action(state)
            elif isinstance(agent, QAC):
                action = agent.select_action(state)
            elif isinstance(agent, (A2C, NaturalAC)):
                action, _, _ = agent.select_action(state)  # No training, so log_prob, value not used here
            elif isinstance(agent, DynaQ):
                action = agent.choose_action(state, training=False)  # Eval with greedy policy
            else:
                raise TypeError(f"Unknown agent type: {type(agent)}")

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
            if steps >= MAX_STEPS_PER_EPISODE:
                truncated = True

        total_rewards.append(episode_reward)
        if episode_reward > 100:
            total_successes += 1

    avg_reward = np.mean(total_rewards)
    success_rate = total_successes / num_episodes
    return avg_reward, success_rate, total_rewards # 新的返回，增加了 total_rewards 列表



# --- Training Function ---
def train_agent(env_name, agent_name, agent_class, agent_params,
                num_episodes, eval_freq, num_test_episodes, seed=42):
    print(f"Training {agent_name}...")
    env = gym.make(env_name)
    eval_env = gym.make(env_name)  # Separate env for evaluation

    # For reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # env.seed(seed) # For older gym
    # eval_env.seed(seed+1)
    env.action_space.seed(seed)
    eval_env.action_space.seed(seed + 1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    current_agent_params = agent_params.copy()
    if agent_name not in ["Dyna-Q"]:  # Assuming Dyna-Q is the only non-torch agent
        current_agent_params['device'] = device  # Add device from global scope

    if agent_name == "Dyna-Q":
        agent = agent_class(state_dim, action_dim, **agent_params)  # Original DynaQ params
    else:
        agent = agent_class(state_dim, action_dim, **current_agent_params)

    eval_rewards_history = []
    eval_success_rates_history = []
    eval_at_episodes = []

    # For moving average of training rewards
    training_rewards_deque = deque(maxlen=100)
    avg_training_rewards_history = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed + episode if seed is not None else None)
        current_episode_reward = 0
        done = False
        truncated = False

        # Specific handling for REINFORCE agent needing to clear memory at start
        if isinstance(agent, REINFORCE):
            agent.clear_memory()

        for step in range(MAX_STEPS_PER_EPISODE):
            # Action selection
            if isinstance(agent, REINFORCE):
                action, log_prob = agent.select_action(state)
            elif isinstance(agent, QAC):
                action = agent.select_action(state)
            elif isinstance(agent, (A2C, NaturalAC)):
                action, log_prob, value = agent.select_action(state)
            elif isinstance(agent, DynaQ):
                action = agent.choose_action(state, training=True)
            else:
                raise TypeError("Unknown agent type for action selection.")

            next_state, reward, done, truncated, _ = env.step(action)
            current_episode_reward += reward

            # Agent update/storage
            if isinstance(agent, REINFORCE):
                agent.store_outcome(log_prob, reward)
            elif isinstance(agent, QAC):
                agent.update(state, action, reward, next_state, done or truncated)
            elif isinstance(agent, (A2C, NaturalAC)):
                agent.store_transition(state, action, log_prob, reward, done or truncated, value)
            elif isinstance(agent, DynaQ):
                agent.update(state, action, reward, next_state, done or truncated)
            else:
                raise TypeError("Unknown agent type for update.")

            state = next_state
            if done or truncated:
                break

        # Episode-end updates for relevant agents
        if isinstance(agent, REINFORCE):
            agent.update()
        elif isinstance(agent, (A2C, NaturalAC)):
            # last_next_state_if_not_terminal is `next_state` if loop broke by truncation, else None if `done` was true
            agent.update(next_state if truncated and not done else None)
        elif isinstance(agent, DynaQ):
            agent.decay_epsilon()

        training_rewards_deque.append(current_episode_reward)
        avg_training_rewards_history.append(np.mean(training_rewards_deque))

        if episode % eval_freq == 0:
            # avg_reward, success_rate = evaluate_agent(eval_env, agent, num_test_episodes)
            avg_reward, success_rate, _ = evaluate_agent(eval_env, agent, num_test_episodes)  # 使用 _ 忽略不需要的返回值

            eval_rewards_history.append(avg_reward)
            eval_success_rates_history.append(success_rate)
            eval_at_episodes.append(episode)
            print(
                f"Agent: {agent_name}, Episode: {episode}, Avg Test Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")

            # Check for convergence for "Convergence Speed" metric
            if success_rate >= 0.8 and not hasattr(agent, 'converged_at_episode'):
                agent.converged_at_episode = episode
                print(f"Agent {agent_name} reached 80% success rate at episode {episode}")

    env.close()
    eval_env.close()

    # Store final performance for box plot (rewards from last evaluation)
    # final_eval_rewards, _ = evaluate_agent(gym.make(env_name), agent,
    #                                        num_test_episodes * 2)  # More samples for box plot
    _, _, final_test_rewards_list = evaluate_agent(gym.make(env_name), agent, num_test_episodes * 2)  # 新的方式

    return {
        "agent_name": agent_name,
        "eval_episodes": eval_at_episodes,
        "eval_rewards": eval_rewards_history,
        "eval_success_rates": eval_success_rates_history,
        "avg_training_rewards": avg_training_rewards_history,  # For smoothed training curve
        "training_episodes_x": list(range(1, num_episodes + 1)),
        "convergence_episode": getattr(agent, 'converged_at_episode', num_episodes + 1),  # Default if not converged
        # "final_test_rewards": final_eval_rewards  # For box plot
        "final_test_rewards": final_test_rewards_list # 新的，这是一个包含多次奖励的列表
    }


# --- Plotting Function ---
def plot_results(all_results):
    plt.style.use('seaborn-darkgrid')  # Using a seaborn style

    # 1. Training Curve (Average Evaluation Reward vs Episodes)
    plt.figure(figsize=(12, 7))
    for result in all_results:
        plt.plot(result["eval_episodes"], result["eval_rewards"], label=result["agent_name"], linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Average Evaluation Reward (10 test runs)")
    plt.title("Training Curve: Average Evaluation Reward")
    plt.legend()
    plt.savefig("plots0.001/training_curve_eval_reward.png")
    plt.show()

    # Plotting smoothed training rewards (optional, but good for seeing training progress)
    plt.figure(figsize=(12, 7))
    for result in all_results:
        plt.plot(result["training_episodes_x"], result["avg_training_rewards"],
                 label=f"{result['agent_name']} (Training Avg)", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Average Training Reward (Moving Avg 100 eps)")
    plt.title("Smoothed Training Rewards")
    plt.legend()
    plt.savefig("plots0.001/training_curve_training_reward.png")
    plt.show()

    # 2. Success Rate Curve
    plt.figure(figsize=(12, 7))
    for result in all_results:
        plt.plot(result["eval_episodes"], result["eval_success_rates"], label=result["agent_name"], linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Success Rate (Reward > 100 in test runs)")
    plt.title("Success Rate During Evaluation")
    plt.axhline(y=0.8, color='gray', linestyle='--', label="80% Success Target")
    plt.legend()
    plt.savefig("plots0.001/success_rate_curve.png")
    plt.show()

    # 3. Final Performance Box Plot
    final_rewards_data = [res["final_test_rewards"] for res in all_results]
    agent_names = [res["agent_name"] for res in all_results]
    plt.figure(figsize=(12, 7))
    plt.boxplot(final_rewards_data, labels=agent_names, patch_artist=True, medianprops={'linewidth': 2})
    plt.ylabel("Reward in Final Test Episodes")
    plt.title("Final Performance Distribution (Box Plot)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("plots0.001/final_performance_boxplot.png")
    plt.show()

    # 4. Convergence Speed
    print("\n--- Convergence Speed (Episodes to reach 80% success rate) ---")
    for result in all_results:
        conv_ep = result["convergence_episode"]
        if conv_ep <= NUM_EPISODES:
            print(f"{result['agent_name']}: {conv_ep} episodes")
        else:
            print(f"{result['agent_name']}: Did not converge to 80% success within {NUM_EPISODES} episodes.")


# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    agents_to_run = [
        {"name": "REINFORCE", "class": REINFORCE, "params": {"learning_rate": LR_NN, "gamma": GAMMA, "device": device}},
        {"name": "QAC", "class": QAC,
         "params": {"actor_learning_rate": LR_NN, "critic_learning_rate": LR_NN, "gamma": GAMMA, "device": device}},
        {"name": "A2C", "class": A2C,
         "params": {"actor_learning_rate": LR_NN, "critic_learning_rate": LR_NN, "gamma": GAMMA,
                    "gae_lambda": GAE_LAMBDA, "device": device}},
        {"name": "NaturalAC", "class": NaturalAC, "params": {
            "actor_learning_rate": LR_NN, "critic_learning_rate": LR_NN, "gamma": GAMMA, "gae_lambda": GAE_LAMBDA,
            "cg_iters": CG_ITERS, "cg_damping": CG_DAMPING, "fim_update_interval": FIM_UPDATE_INTERVAL,
            "max_kl_step": MAX_KL_STEP, "device": device
        }
         },
        {"name": "Dyna-Q", "class": DynaQ, "params": {  # Dyna-Q does not use torch device
            "bins_per_dim": DYNQ_BINS, "state_lows": STATE_LOWS, "state_highs": STATE_HIGHS,
            "alpha": LR_DYNQ_ALPHA, "gamma": GAMMA,
            "epsilon_start": EPSILON_START, "epsilon_min": EPSILON_MIN,
            "epsilon_decay_episodes": EPSILON_DECAY_EPISODES,
            "planning_steps": PLANNING_STEPS
        }
         },
    ]

    all_experiment_results = []

    for agent_config in agents_to_run:
        results = train_agent(
            ENV_NAME,
            agent_config["name"],
            agent_config["class"],
            agent_config["params"],
            NUM_EPISODES,
            EVAL_FREQ,
            NUM_TEST_EPISODES
        )
        all_experiment_results.append(results)
        print(f"Finished training {agent_config['name']}.\n")

    print("All training finished. Plotting results...")
    plot_results(all_experiment_results)
    print("Plots saved. Experiment complete.")