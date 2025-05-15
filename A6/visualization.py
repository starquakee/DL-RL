import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import copy
import matplotlib as mpl
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_value_function_convergence(V_history_list, labels, title="Value Function Convergence (RMSE)",
                                    reference_V=None, env_size=None, all_states_indices=None):
    """
    Plots the Root Mean Squared Error (RMSE) of value functions over episodes/iterations.
    RMSE is calculated against a reference_V if provided, or against the final V in the history
    for each algorithm.

    Args:
        V_history_list: A list of V_history series. Each V_history is a list of V dictionaries
                        (defaultdict(float) where keys are state_indices) recorded at intervals.
        labels: A list of labels for each V_history series.
        title: The title of the plot.
        reference_V: (Optional) A reference V (defaultdict or dict) to compute RMSE against.
                     If None, RMSE is computed against the final V of that algorithm's history.
        env_size: (Optional) Tuple (rows, cols) or int for square grid. Used if all_states_indices is None.
        all_states_indices: (Optional) A list of all valid (non-obstacle, non-terminal) state indices to consider for RMSE.
                           If None, it attempts to infer from the keys of V_history or reference_V.
    """
    plt.figure(figsize=(12, 7))

    for i, V_history in enumerate(V_history_list):
        if not V_history:
            print(f"Warning: V_history for label '{labels[i]}' is empty. Skipping.")
            continue

        num_snapshots = len(V_history)
        rmse_values = []

        final_V_for_algo = V_history[-1]

        active_states = None
        if all_states_indices:
            active_states = all_states_indices
        elif reference_V:
            active_states = list(reference_V.keys())
        elif final_V_for_algo:
            active_states = list(final_V_for_algo.keys())

        if not active_states:
            print(f"Warning: Could not determine active states for RMSE for '{labels[i]}'. Skipping.")
            continue

        for v_snapshot in V_history:
            if reference_V:
                target_v = reference_V
            else:
                target_v = final_V_for_algo

            squared_errors = []
            for state_idx in active_states:
                val_snapshot = v_snapshot.get(state_idx, 0.0)
                val_target = target_v.get(state_idx, 0.0)
                squared_errors.append((val_snapshot - val_target) ** 2)

            if squared_errors:
                rmse = np.sqrt(np.mean(squared_errors))
                rmse_values.append(rmse)
            else:
                rmse_values.append(0)

        episode_ticks = np.linspace(0, 1, num_snapshots) * (len(V_history))  # Placeholder

        plt.plot(rmse_values, label=labels[i])

    plt.xlabel("Evaluation Points (e.g., Episodes / Interval)")
    plt.ylabel("RMSE")
    if reference_V:
        plt.title(title + " (vs Reference V)")
    else:
        plt.title(title + " (vs Own Final V)")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_value_function(V, env, title="State-Value Function"):
    """
    Visualizes the state-value function as a heatmap on the grid.
    Args:
        V: A dictionary (or defaultdict) mapping state_index to value.
        env: The GridWorld environment instance (to get size, obstacles, terminals).
        title: The title of the plot.
    """
    grid_values = np.zeros((env.size, env.size))
    annotation = np.empty((env.size, env.size), dtype=object)

    for r in range(env.size):
        for c in range(env.size):
            state_idx = env.get_state_index((r, c))
            if env.is_obstacle(r, c):
                grid_values[r, c] = -np.inf  # Or a very small number to distinguish
                annotation[r, c] = "WALL"
            elif env.is_terminal_state[(r, c)]:
                grid_values[r, c] = V.get(state_idx, env.rewards[(r, c)])  # Show actual reward for terminal
                if (r, c) == env.goal_pos:
                    annotation[r, c] = f"GOAL\n{grid_values[r, c]:.1f}"
                else:  # Trap
                    annotation[r, c] = f"TRAP\n{grid_values[r, c]:.1f}"
            else:
                grid_values[r, c] = V.get(state_idx, 0)  # Default to 0 if state not in V
                annotation[r, c] = f"{grid_values[r, c]:.2f}"

    fig, ax = plt.subplots(figsize=(env.size + 2, env.size + 2))

    # Create a masked array for obstacles to color them differently
    masked_values = np.ma.masked_where(grid_values == -np.inf, grid_values)

    original_cmap = mpl.cm.get_cmap("viridis")
    cmap = copy.copy(original_cmap)  # 使用 copy.copy() 进行复制
    cmap.set_bad(color='grey')

    im = ax.imshow(masked_values, cmap=cmap, interpolation='nearest')

    # Add text annotations
    for r in range(env.size):
        for c in range(env.size):
            text_color = "black"
            ax.text(c, r, annotation[r, c], ha="center", va="center", color=text_color, fontsize=8)

    ax.set_title(title)
    ax.set_xticks(np.arange(env.size))
    ax.set_yticks(np.arange(env.size))
    ax.set_xticklabels(np.arange(env.size))
    ax.set_yticklabels(np.arange(env.size))
    plt.colorbar(im, ax=ax, orientation='vertical', label='State Value')
    plt.tight_layout()
    plt.show()


def visualize_eligibility_traces(E, env, title="Eligibility Traces (E)", episode_num=None, step_num=None):
    """
    Visualizes the eligibility traces as a heatmap on the grid.
    Args:
        E: A dictionary (or defaultdict) mapping state_index to trace value.
        env: The GridWorld environment instance.
        title: The title of the plot.
        episode_num: (Optional) Episode number for the title.
        step_num: (Optional) Step number for the title.
    """
    grid_traces = np.zeros((env.size, env.size))
    max_trace = 0
    if E:
        valid_traces = [val for val in E.values() if isinstance(val, (int, float))]
        if valid_traces:
            max_trace = max(valid_traces) if valid_traces else 0
        else:  # E might contain non-numeric if used incorrectly, or be empty
            max_trace = 0

    for r in range(env.size):
        for c in range(env.size):
            state_idx = env.get_state_index((r, c))
            if env.is_obstacle(r, c):
                grid_traces[r, c] = -np.inf  # Mask obstacles
            else:
                grid_traces[r, c] = E.get(state_idx, 0.0)

    full_title = title
    if episode_num is not None: full_title += f" - Episode {episode_num}"
    if step_num is not None: full_title += f", Step {step_num}"

    fig, ax = plt.subplots(figsize=(env.size + 1, env.size + 1))
    masked_traces = np.ma.masked_where(grid_traces == -np.inf, grid_traces)

    original_cmap = mpl.cm.get_cmap("plasma")
    cmap = copy.copy(original_cmap)  # 使用 copy.copy() 进行复制
    cmap.set_bad(color='lightgrey')

    # Normalize color scale if max_trace is very small or zero
    vmin_plot, vmax_plot = 0, max(max_trace, 1e-9)  # Avoid division by zero if max_trace is 0

    im = ax.imshow(masked_traces, cmap=cmap, interpolation='nearest', vmin=vmin_plot, vmax=vmax_plot)

    for r in range(env.size):
        for c in range(env.size):
            if not env.is_obstacle(r, c) and not env.is_terminal_state[(r, c)]:
                trace_val = grid_traces[r, c]
                if not np.ma.is_masked(trace_val):
                    ax.text(c, r, f"{trace_val:.2f}", ha="center", va="center",
                            color="white" if trace_val > vmax_plot * 0.6 else "black", fontsize=7)
            elif env.is_obstacle(r, c):
                ax.text(c, r, "WALL", ha="center", va="center", color="black", fontsize=7)
            elif env.is_terminal_state[(r, c)]:
                ax.text(c, r, "TERM", ha="center", va="center", color="black", fontsize=7)

    ax.set_title(full_title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation='vertical', label='Trace Value E(s)')
    plt.tight_layout()
    plt.show()


def plot_parameter_sensitivity(results, param_name, metric_name="Final RMSE", title=None):
    """
    Plots the sensitivity of an algorithm to a specific parameter.
    Args:
        results: A list of tuples, where each tuple is (param_value, metric_value).
                 Example: [(0.1, 0.5), (0.2, 0.4), ...] for (alpha, RMSE)
        param_name: Name of the parameter (e.g., "Alpha", "Lambda").
        metric_name: Name of the metric (e.g., "Final RMSE", "Average Reward").
        title: (Optional) Plot title.
    """
    if not results:
        print("No results to plot for parameter sensitivity.")
        return

    results.sort(key=lambda x: x[0])
    param_values = [r[0] for r in results]
    metric_values = [r[1] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, metric_values, marker='o', linestyle='-')

    plt.xlabel(param_name)
    plt.ylabel(metric_name)
    if title:
        plt.title(title)
    else:
        plt.title(f"{metric_name} vs. {param_name}")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    print("Testing visualization.py...")


    # 1. Dummy Environment
    class DummyEnvForViz:
        def __init__(self, size=5):
            self.size = size
            self.rewards = np.zeros((size, size))
            self.is_terminal_state = np.full((size, size), False, dtype=bool)
            self.obstacles_pos = [(1, 1), (2, 3)]
            self.goal_pos = (size - 1, size - 1)
            self.rewards[self.goal_pos] = 10
            self.is_terminal_state[self.goal_pos] = True
            self.trap_pos = (0, size - 1)
            self.rewards[self.trap_pos] = -10
            self.is_terminal_state[self.trap_pos] = True

        def get_state_index(self, state_pos):
            return state_pos[0] * self.size + state_pos[1]

        def is_obstacle(self, r, c):
            return (r, c) in self.obstacles_pos

        def get_all_states_indices_values(self):  # For RMSE calculation
            indices = []
            for r in range(self.size):
                for c in range(self.size):
                    if not self.is_obstacle(r, c) and not self.is_terminal_state[(r, c)]:
                        indices.append(self.get_state_index((r, c)))
            return indices


    env = DummyEnvForViz(size=5)
    all_state_indices = env.get_all_states_indices_values()

    V_hist1 = []
    V_hist2 = []
    ref_V = defaultdict(float)
    for r in range(env.size):
        for c in range(env.size):
            idx = env.get_state_index((r, c))
            if not env.is_obstacle(r, c) and not env.is_terminal_state[(r, c)]:
                ref_V[idx] = np.random.rand() * 5  # Dummy reference values

    for i in range(20):  # 20 snapshots
        v_snap1 = defaultdict(float)
        v_snap2 = defaultdict(float)
        for idx in all_state_indices:
            # Simulate convergence: error decreases over time
            v_snap1[idx] = ref_V[idx] + np.random.rand() * (1.0 / (i + 1))
            v_snap2[idx] = ref_V[idx] + np.random.rand() * (1.5 / (i + 1))  # Algo 2 converges slower
        V_hist1.append(dict(v_snap1))
        V_hist2.append(dict(v_snap2))

    # Ensure terminal states are in V_hist for consistent key sets if not using all_state_indices carefully
    for r_idx_term in range(env.size):
        for c_idx_term in range(env.size):
            if env.is_terminal_state[(r_idx_term, c_idx_term)]:
                s_idx_term = env.get_state_index((r_idx_term, c_idx_term))
                for v_s in V_hist1: v_s[s_idx_term] = 0.0
                for v_s in V_hist2: v_s[s_idx_term] = 0.0
                ref_V[s_idx_term] = 0.0

    plot_value_function_convergence(
        [V_hist1, V_hist2],
        labels=["Algorithm 1", "Algorithm 2"],
        reference_V=ref_V,
        all_states_indices=all_state_indices
    )
    plot_value_function_convergence(
        [V_hist1, V_hist2],
        labels=["Algorithm 1 (vs own final)", "Algorithm 2 (vs own final)"],
        all_states_indices=all_state_indices
    )

    # 3. Dummy Data for Value Function Visualization
    final_V_algo1 = V_hist1[-1]
    visualize_value_function(final_V_algo1, env, title="Final Value Function (Algo 1)")

    # 4. Dummy Data for Eligibility Traces Visualization
    dummy_E = defaultdict(float)
    # Simulate some traces after a few steps in an episode
    dummy_E[env.get_state_index((0, 0))] = 1.0
    dummy_E[env.get_state_index((0, 1))] = 0.9
    dummy_E[env.get_state_index((0, 2))] = 0.81
    dummy_E[env.get_state_index((1, 2))] = 0.72
    if env.is_obstacle(1, 1):
        try:
            env.obstacles_pos.remove((1, 1))
        except ValueError:
            pass

    visualize_eligibility_traces(dummy_E, env, title="Sample Eligibility Traces", episode_num=5, step_num=10)

    # 5. Dummy Data for Parameter Sensitivity Plot
    alpha_results = [
        (0.01, 0.8), (0.05, 0.5), (0.1, 0.3), (0.2, 0.35), (0.5, 0.6)
    ]
    plot_parameter_sensitivity(alpha_results, "Learning Rate (Alpha)", "Final RMSE", "Alpha Sensitivity for TD(0)")

    lambda_results = [
        (0.0, 0.4), (0.25, 0.3), (0.5, 0.2), (0.75, 0.22), (0.9, 0.25), (1.0, 0.35)
    ]
    plot_parameter_sensitivity(lambda_results, "Lambda", "Final RMSE", "Lambda Sensitivity for TD(Lambda)")

    print("Visualization tests completed. Check for plots.")
