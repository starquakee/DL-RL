import matplotlib.pyplot as plt
import numpy as np

# Load the saved rolling average reward data for each algorithm
try:
    mc_avg_rewards = np.load('monte_carlo_avg_rewards.npy')  # Assuming this was saved from MC script
    sarsa_avg_rewards = np.load('sarsa_avg_rewards.npy')  # Assuming this was saved from SARSA script
    # Ensure your SARSA-Lambda filename matches what was saved
    sarsa_lambda_avg_rewards = np.load('sarsa_lambda_avg_rewards_lambda0.9_replacing.npy')
    q_learning_avg_rewards = np.load(
        'q_learning_rolling_avg_rewards.npy')  # Corrected filename based on previous context
    dqn_avg_rewards = np.load('dqn_rolling_avg_scores.npy')  # This should be the rolling average, not raw scores
except FileNotFoundError as e:
    print(
        f"Error: One or more reward files not found. Please ensure all algorithms have been trained and their rolling average reward data saved correctly.")
    print(f"Missing file detail: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading files: {e}")
    exit()

# Create x-axis data (episode numbers) for each algorithm
# The length of each array corresponds to the number of episodes trained
episodes_mc = np.arange(1, len(mc_avg_rewards) + 1)
episodes_sarsa = np.arange(1, len(sarsa_avg_rewards) + 1)
episodes_sarsa_lambda = np.arange(1, len(sarsa_lambda_avg_rewards) + 1)
episodes_q = np.arange(1, len(q_learning_avg_rewards) + 1)
episodes_dqn = np.arange(1, len(dqn_avg_rewards) + 1)

# --- Plotting ---
plt.figure(figsize=(14, 8))

plt.plot(episodes_mc, mc_avg_rewards, label='Monte Carlo')
plt.plot(episodes_sarsa, sarsa_avg_rewards, label='SARSA')
plt.plot(episodes_sarsa_lambda, sarsa_lambda_avg_rewards, label='SARSA(Lambda=0.9)')
plt.plot(episodes_q, q_learning_avg_rewards, label='Q-learning')
plt.plot(episodes_dqn, dqn_avg_rewards, label='DQN')

# --- English Labels and Title ---
plt.xlabel('Training Episodes')
plt.ylabel('Average Reward over last 100 Episodes')
plt.title('Training Curve Comparison of Reinforcement Learning Algorithms')
plt.legend()
plt.grid(True)

# --- X-axis Ticks every 2500 episodes ---
# Determine the overall maximum number of episodes to set the x-axis limit and ticks
all_max_episodes = [len(mc_avg_rewards), len(sarsa_avg_rewards), len(sarsa_lambda_avg_rewards),
                    len(q_learning_avg_rewards), len(dqn_avg_rewards)]
max_ep_val = 0
if all_max_episodes:  # Check if the list is not empty
    max_ep_val = max(all_max_episodes)

if max_ep_val > 0:
    plt.xlim(0, max_ep_val + 1)  # Set x-axis limit slightly beyond the max episodes
    tick_interval = 2500
    # Ensure ticks start from 0 or the first interval mark and go up to or slightly beyond max_ep_val
    num_intervals = int(np.ceil(max_ep_val / tick_interval))
    xticks = np.arange(0, (num_intervals + 1) * tick_interval, tick_interval)
    # Filter ticks to be within a reasonable range if max_ep_val is small, or adjust as needed
    # For example, if max_ep_val is 2000, you might not want a tick at 2500 yet.
    # The following ensures ticks don't go excessively beyond max_ep_val if max_ep_val is not a multiple of tick_interval
    xticks = xticks[
        xticks <= max_ep_val + (tick_interval - max_ep_val % tick_interval if max_ep_val % tick_interval != 0 else 0)]
    if xticks[0] == 0 and len(xticks) > 1 and xticks[1] > max_ep_val:  # handles cases where max_ep_val < tick_interval
        xticks = np.array([0, max_ep_val]) if max_ep_val > 0 else np.array([0])
    elif max_ep_val > 0 and xticks[0] != 0:  # If max_ep_val is small and first tick is >0
        xticks = np.concatenate(([0], xticks))

    # A simpler way if you want ticks strictly up to max_ep_val and including 0
    # xticks = np.arange(0, max_ep_val + tick_interval, tick_interval)
    # xticks = xticks[xticks <= max_ep_val]
    # if 0 not in xticks and max_ep_val > 0:
    #    xticks = np.insert(xticks, 0, 0)
    # if max_ep_val not in xticks and max_ep_val > 0:
    #    xticks = np.append(xticks, max_ep_val)
    # xticks = np.unique(xticks) # Ensure sorted unique ticks

    # Using a more direct approach for ticks:
    if max_ep_val > 0:
        custom_ticks = list(np.arange(0, max_ep_val + 1, tick_interval))
        if max_ep_val not in custom_ticks:  # Ensure the last point is a tick if it's not a multiple
            # Add it only if it's meaningfully far from the last tick
            if not custom_ticks or max_ep_val > custom_ticks[-1] + tick_interval / 4:
                custom_ticks.append(max_ep_val)
        custom_ticks = sorted(list(set(custom_ticks)))  # Ensure unique and sorted
        plt.xticks(custom_ticks)

plt.tight_layout()  # Adjust layout to prevent labels from overlapping
plt.savefig('all_algorithms_training_comparison.png')
plt.show()
